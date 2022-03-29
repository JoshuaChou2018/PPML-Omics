# -*- coding: utf-8 -*-
# @Time    : 2021-09-30 23:30
# @Author  : Juexiao Zhou & Siyuan Chen

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from scipy import stats
from tqdm import tqdm
import copy
from math import sqrt, exp
from scipy.special import erf
import logging
import time
import pandas as pd
import argparse
from multiprocessing import Pool
import random
from sklearn.cluster import KMeans
import metrics as M
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
root = os.path.dirname(__file__)

class RunningAverage():
    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)

def log_creater(output_dir,expname):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    log_name='{}.log'.format(expname)
    #log_name = mode+'_{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
    final_log_file = os.path.join(output_dir,log_name)

    # creat a log
    log = logging.getLogger('train_log')
    log.setLevel(logging.DEBUG)

    # FileHandler
    file = logging.FileHandler(final_log_file,'w')
    file.setLevel(logging.DEBUG)

    # StreamHandler
    stream = logging.StreamHandler()
    stream.setLevel(logging.DEBUG)

    # Formatter
    formatter = logging.Formatter(
        '[%(asctime)s][line: %(lineno)d] ==> [INFO] %(message)s')

    # setFormatter
    file.setFormatter(formatter)
    stream.setFormatter(formatter)

    # addHandler
    log.addHandler(file)
    log.addHandler(stream)

    log.info('creating {}'.format(final_log_file))
    return log


def save_checkpoint(model, is_best, checkpoint,log, expname):

    #model_name = os.path.join(checkpoint, 'mode_{}_'.format(mode)+'epoch_' + str(epoch) + '_model.pth.tar')
    model_name=os.path.join(checkpoint, '{}'.format(expname) + '_modelbest.tar')

    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)

    if is_best:
        log.info("Saving checkpoint... ")
        log.info("Saved to {}".format(model_name))
        log.info("Checkpoint is best!")
        with open(model_name, 'wb') as f:
            torch.save(model.state_dict(), f)
    else:
        log.info("Model not good, skip!")

class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model,self).__init__()

        self.hidden1 = nn.Linear(input_size,1024)
        self.hidden2 = nn.Linear(1024, 128)
        self.hidden3 = nn.Linear(128, 1024)
        self.hidden4 = nn.Linear(1024, output_size)  # output layer

    def forward(self, x):
        x = F.relu(self.hidden1(x))  # activation function for hidden layer
        x = F.relu(self.hidden2(x))
        feat = x
        x = F.relu(self.hidden3(x))
        x = self.hidden4(x)  # linear output

        return feat, x


class Single_Cell_Data(Dataset):
    def __init__(self, dataset_name):

        count_file = os.path.join(root,"data", "count_{}.csv".format(dataset_name))
        label_file = os.path.join(root,"data", "label_{}.csv".format(dataset_name))
        print("Reading {}".format(count_file))

        if os.path.exists(count_file) == False:
            raise Exception("{} doesn't exist".format(count_file))

        feature = np.array(pd.read_csv(count_file))
        label = np.array(pd.read_csv(label_file))[:,1].tolist()

        self.label_dic = {}
        for i, l in enumerate(set(label)):
            self.label_dic[l] = i
        self.label_dic = dict(sorted(self.label_dic.items()))
        print(self.label_dic)

        self.feature_size = feature.shape[1]
        self.samples = []
        print("{} samples".format(feature.shape[0]))
        for i in range(feature.shape[0]):
            norm = np.linalg.norm(feature[i])
            gene = feature[i] / norm
            #gene = np.log10(feature[i]+0.01)
            info = {
                "genes": gene,
                "celltype": self.label_dic[label[i]]
            }
            self.samples.append(info)


    def __getitem__(self, idx):
        data_dic = copy.deepcopy(self.samples[idx])
        genes = data_dic['genes']
        label=  data_dic['celltype']

        return torch.Tensor(genes), label

    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small batch size of sets.
        return len(self.samples)

def initialClientModel(serverModel,device):
    clientModel=Model(input_size=input_size,output_size=input_size)
    clientModel=clientModel.to(device=device)
    clientModel = torch.nn.DataParallel(clientModel).cuda()
    clientModel.load_state_dict(serverModel.state_dict())
    return clientModel

def compute_grad_update(old_model, new_model, lr, device):
    # maybe later to implement on selected layers/parameters
    if device:
        old_model, new_model = old_model.to(device), new_model.to(device)
    return [(new_param.data - old_param.data)/(-lr) for old_param, new_param in zip(old_model.parameters(), new_model.parameters())]

def l2norm(grad):
    return torch.sum(torch.pow(flatten(grad), 2))

def add_gradient_updates(grad_update_1, grad_update_2, weight=1.0):
    for param_1, param_2 in zip(grad_update_1, grad_update_2):
        param_1.data += param_2.data * weight

def calibrateAnalyticGaussianMechanism(epsilon, delta, GS=1, tol=1.e-12):

    """ Calibrate a Gaussian perturbation for differential privacy using the analytic Gaussian mechanism of [Balle and Wang, ICML'18]
        Arguments:
        epsilon : target epsilon (epsilon > 0)
        delta : target delta (0 < delta < 1)
        GS : upper bound on L2 global sensitivity (GS >= 0)
        tol : error tolerance for binary search (tol > 0)
        Output:
        sigma : standard deviation of Gaussian noise needed to achieve (epsilon,delta)-DP under global sensitivity GS
    """

    global shuffle_model
    global numberClients
    if shuffle_model==True:
        epsilon=epsilon*numberClients

    def Phi(t):
        return 0.5 * (1.0 + erf(float(t) / sqrt(2.0)))

    def caseA(epsilon, s):
        return Phi(sqrt(epsilon * s)) - exp(epsilon) * Phi(-sqrt(epsilon * (s + 2.0)))

    def caseB(epsilon, s):
        return Phi(-sqrt(epsilon * s)) - exp(epsilon) * Phi(-sqrt(epsilon * (s + 2.0)))

    def doubling_trick(predicate_stop, s_inf, s_sup):
        while (not predicate_stop(s_sup)):
            s_inf = s_sup
            s_sup = 2.0 * s_inf
        return s_inf, s_sup

    def binary_search(predicate_stop, predicate_left, s_inf, s_sup):
        s_mid = s_inf + (s_sup - s_inf) / 2.0
        while (not predicate_stop(s_mid)):
            if (predicate_left(s_mid)):
                s_sup = s_mid
            else:
                s_inf = s_mid
            s_mid = s_inf + (s_sup - s_inf) / 2.0
        return s_mid

    delta_thr = caseA(epsilon, 0.0)

    if (delta == delta_thr):
        alpha = 1.0

    else:
        if (delta > delta_thr):
            predicate_stop_DT = lambda s: caseA(epsilon, s) >= delta
            function_s_to_delta = lambda s: caseA(epsilon, s)
            predicate_left_BS = lambda s: function_s_to_delta(s) > delta
            function_s_to_alpha = lambda s: sqrt(1.0 + s / 2.0) - sqrt(s / 2.0)

        else:
            predicate_stop_DT = lambda s: caseB(epsilon, s) <= delta
            function_s_to_delta = lambda s: caseB(epsilon, s)
            predicate_left_BS = lambda s: function_s_to_delta(s) < delta
            function_s_to_alpha = lambda s: sqrt(1.0 + s / 2.0) + sqrt(s / 2.0)

        predicate_stop_BS = lambda s: abs(function_s_to_delta(s) - delta) <= tol

        s_inf, s_sup = doubling_trick(predicate_stop_DT, 0.0, 1.0)
        s_final = binary_search(predicate_stop_BS, predicate_left_BS, s_inf, s_sup)
        alpha = function_s_to_alpha(s_final)

    sigma = alpha * GS / sqrt(2.0 * epsilon)
    return sigma

def sign(grad):
    return torch.sign(grad)

def flatten(grad):
    return grad.data.view(-1)

def dpvalue(p):
  return torch.tensor(1) if np.random.random() < p else torch.tensor(-1)

def Phi(t):
    return 0.5*(1.0 + erf(float(t)/sqrt(2.0)))

def parallel_dpvalue(p):
    return dpvalue(Phi(p))

def run_parallel_dpvalue(ps):
    global nprocess
    with Pool(processes=nprocess) as pool:
        processed_data=pool.map(parallel_dpvalue,ps)
    return processed_data

def dpsign(grad, device,epsilon=8, delta=1e-3):
    global l2_norm_clip
    sigma=calibrateAnalyticGaussianMechanism(epsilon=epsilon, delta=delta,GS=l2_norm_clip)
    #print(grad)
    grad=torch.tensor(grad).to(device)
    grad_sigma=grad/torch.tensor([sigma]).to(device)
    grad_phi=torch.tensor([0.5]).to(device)*(torch.tensor([1.0]).to(device) + torch.erf(grad_sigma/sqrt(torch.tensor([2.0]).to(device))))
    random_matrix=torch.tensor(np.random.random(size=grad.shape),device=device)
    bool_matrix=(grad_phi>random_matrix).type(torch.uint8)
    bool_matrix[bool_matrix==0]=-1
    result=bool_matrix
    #result=torch.tensor([dpvalue(Phi(i/sigma)) for i in flatten(grad)],device=device).reshape(grad.shape)
    #result = sign(torch.tensor(np.random.normal(0, sigma, grad.shape), device=device) + grad)
    return result

def dp(grad, device,epsilon=8, delta=1e-3):
    global l2_norm_clip
    sigma = calibrateAnalyticGaussianMechanism(epsilon=epsilon, delta=delta, GS=l2_norm_clip)
    #sigma = 0.0001 #TODO
    result = torch.tensor(np.random.normal(0, sigma, grad.shape), device=device) + grad
    #result = torch.tensor([torch.tensor(np.random.normal(0, sigma,len(flatten(grad))), device=device) + flatten(grad)], device=device).reshape(grad.shape)
    return result

def add_update_to_model(model, update, weight=1.0, device=None):
    if not update: return model
    if device:
        model = model.to(device)
        update = [param.to(device) for param in update]

    for param_model, param_update in zip(model.parameters(), update):
        param_model.data += weight * param_update.data
    return model

def updateServerModel(server_model, grad_updates, lr, device=None, mode='SGD',epsilon=8, delta=1e-3):
    aggregated=[]
    if mode == 'SIGNSGD':
        for i in range(len(grad_updates)):
            aggregated.append(sign(grad_updates[i]))
    if mode == 'DPSIGNSGD':
        for i in range(len(grad_updates)):
            aggregated.append(dpsign(grad_updates[i], device=device ,epsilon=epsilon, delta=delta))
    if mode=='DP':
        for i in range(len(grad_updates)):
            aggregated.append(dp(grad_updates[i], device=device ,epsilon=epsilon, delta=delta))
    if mode=='SGD':
        aggregated=grad_updates

    add_update_to_model(server_model, aggregated, weight=-1.0 * lr)
    return server_model

def make_optimizer_class(cls):
    class DPOptimizerClass(cls):
        def __init__(self, l2_norm_clip, **kwargs):
            super(DPOptimizerClass, self).__init__(**kwargs)

            self.l2_norm_clip = l2_norm_clip

            for group in self.param_groups:
                group['accum_grads'] = [torch.zeros_like(param.data) if param.requires_grad else None for param in group['params']]

        def zero_grad(self):
            for group in self.param_groups:
                for accum_grad in group['accum_grads']:
                    if accum_grad is not None:
                        accum_grad.zero_()

        def step(self, **kwargs):
            total_norm = 0.
            for group in self.param_groups:
                for param in group['params']:
                    if param.requires_grad:
                        total_norm += param.grad.data.norm(2).item() ** 2.
            total_norm = total_norm ** .5
            clip_coef = min(self.l2_norm_clip / (total_norm + 1e-6), 1.)

            for group in self.param_groups:
                for param, accum_grad in zip(group['params'], group['accum_grads']):
                    if param.requires_grad:
                        accum_grad.add_(param.grad.data.mul(clip_coef))

            for group in self.param_groups:
                for param, accum_grad in zip(group['params'], group['accum_grads']):
                    if param.requires_grad:
                        param.grad.data = accum_grad.clone()
                        # param.grad.data.add_(self.l2_norm_clip * self.noise_multiplier * torch.randn_like(param.grad.data))
                        # param.grad.data.mul_(self.microbatch_size / self.minibatch_size)
            super(DPOptimizerClass, self).step(**kwargs)

    return DPOptimizerClass

DPSGD = make_optimizer_class(torch.optim.SGD)



def Train(logger,
          dataset_name,
          trainLoaders,
          serverModel,
          criterion,
          device,
          num_epochs=25,
          model_path="model",
          mode='SGD'):
    global k
    global lr
    global epsilon
    global delta
    global numberClients
    global l2_norm_clip
    global shuffle_model


    # best_ARI = 0
    best_loss = 99
    is_best = False

    #optimizer = torch.optim.Adam(serverModel.parameters(), lr=lr)

    for epoch in range(num_epochs):

        loss_avg = RunningAverage()

        logger.info('Epoch {}/{}'.format(epoch+1, num_epochs))

        # features = []
        if shuffle_model==False:

            for client_idx in range(numberClients):
                logger.info('client id {}'.format(client_idx))
                trainLoader=trainLoaders[client_idx]

                # initial clientModel
                clientModel = initialClientModel(serverModel, device=device)
                if mode=='SGD':
                    optimizer = torch.optim.SGD(clientModel.parameters(), lr=lr)
                else:
                    torch.nn.utils.clip_grad_norm_(clientModel.parameters(), l2_norm_clip, norm_type=2)
                    optimizer = torch.optim.SGD(clientModel.parameters(), lr=lr)
                clientModel.train()

                with tqdm(total=len(trainLoader)) as t:
                    for genes, label in tqdm(trainLoader):
                        genes, label = genes.to(device), label.to(device)

                        #serverModel.zero_grad()
                        #pred = serverModel(genes)
                        clientModel.zero_grad()

                        feat, out  = clientModel(genes)

                        loss = criterion(genes, out.float())
                        loss.backward()
                        optimizer.step()
                        loss_avg.update(loss.item())
                        # statistics

                        # features += list(feat.detach().cpu().numpy())
                        t.set_postfix(
                                      loss_avg='{:05.10f}'.format(loss_avg()),
                                      total_loss='{:05.10f}'.format(loss.item()),
                                      )
                        t.update()

                #serverModel=clientModel
                # calculate grad update
                grads=compute_grad_update(serverModel, clientModel,lr,device)

                # update to serverModel
                serverModel = updateServerModel(serverModel,
                                                grads,
                                                mode=mode,
                                                lr=lr,
                                                device=device,
                                                epsilon=epsilon,
                                                delta=delta)

        elif shuffle_model==True:

            client_Model_list = []

            for client_idx in range(numberClients):
                logger.info('client id {}'.format(client_idx))
                trainLoader = trainLoaders[client_idx]

                # initial clientModel
                clientModel = initialClientModel(serverModel, device=device)
                if mode == 'SGD':
                    optimizer = torch.optim.SGD(clientModel.parameters(), lr=lr)
                else:
                    torch.nn.utils.clip_grad_norm_(clientModel.parameters(), l2_norm_clip, norm_type=2)
                    optimizer = torch.optim.SGD(clientModel.parameters(), lr=lr)
                    # optimizer = DPSGD(l2_norm_clip=l2_norm_clip,params=clientModel.parameters(),lr=lr)
                clientModel.train()

                with tqdm(total=len(trainLoader)) as t:
                    for genes, label in tqdm(trainLoader):
                        genes, label = genes.to(device), label.to(device)
                        # serverModel.zero_grad()
                        # pred = serverModel(genes)
                        clientModel.zero_grad()
                        feat, out = clientModel(genes)

                        loss = criterion(genes, out.float())
                        loss.backward()
                        optimizer.step()
                        loss_avg.update(loss.item())
                        # statistics
                        t.set_postfix(
                            loss_avg='{:05.3f}'.format(loss_avg()),
                            total_loss='{:05.3f}'.format(loss.item()),
                        )
                        t.update()
                client_Model_list.append(clientModel)
                logger.info("Length of model list: {}".format(len(client_Model_list)))

            # Shuffle the client_Model_list

            logger.info("=> Shuffling the client model")
            random.shuffle(client_Model_list)

            for clientModel in client_Model_list:
                grads = compute_grad_update(serverModel, clientModel, lr, device)
                # update to serverModel
                serverModel = updateServerModel(serverModel, grads, mode=mode, lr=lr, device=device, epsilon=epsilon,
                                                delta=delta)

        features = []
        labels = []
        serverModel.eval()
        print("=> Merging Features ...")
        for client_idx in range(numberClients):
            trainLoader = trainLoaders[client_idx]
            for genes, label in tqdm(trainLoader):
                feat, out = serverModel(genes)
                features += list(feat.detach().cpu().numpy())
                labels += list(label.detach().cpu().numpy())
        print("=> K Means Clustering ...")
        kmeans = KMeans(n_clusters=k, random_state=0).fit(np.array(features))

        pred = kmeans.labels_

        metrics = M.compute_metrics(labels, pred)
        print("{}: \n {}".format(dataset_name, metrics))
        logger.info("Epoch Loss:{}".format(loss_avg()))
        # if  (metrics["ARI"] >= best_ARI):
        if loss_avg() < best_loss:
            is_best = True
            # best_ARI = metrics["ARI"]
            best_loss = loss_avg()
            output = {
                "feature": np.array(features),
                "label": labels,
                "pred": pred
            }
            logger.info('Epoch {} ARI:{} \t NMI:{} \t CA:{} \t JI:{}'.format(epoch,
                                                                             metrics['ARI'],
                                                                             metrics['NMI'],
                                                                             metrics['CA'],
                                                                             metrics['JI'], ))

            save_checkpoint(serverModel, is_best, model_path, logger,
                            expname=expname + "_" + dataset_name+"_epsilon_{}".format(epsilon))

        else:
            logger.info('Epoch {} ARI:{} \t NMI:{} \t CA:{} \t JI:{}'.format(epoch,
                                                                             metrics['ARI'],
                                                                             metrics['NMI'],
                                                                             metrics['CA'],
                                                                             metrics['JI'], ))

    return serverModel, output


def getSingleCellDataset(numClients, dataset_name):

    global logger
    global k
    global input_size
    global label_dic

    trainDataset = Single_Cell_Data(dataset_name)
    k = len(trainDataset.label_dic.keys())
    input_size = trainDataset.feature_size
    label_dic = trainDataset.label_dic
    numItems = np.int(np.floor(len(trainDataset) / numClients))

    trainDatasets=[]
    if numClients!=1:
        if len(trainDataset) % numClients !=0:

            for dataset in torch.utils.data.random_split(trainDataset,
                                                    [numItems for i in range(numClients)] + [len(trainDataset) - numItems * numClients]):
                trainDatasets.append(dataset)

        else:
            for dataset in  torch.utils.data.random_split(trainDataset,
                                                    [numItems for i in range(numClients)]):
                trainDatasets.append(dataset)
    else:
        trainDatasets.append(trainDataset)

    return trainDatasets

if __name__ == '__main__':

    # python FLDP_CC_simulation_App.py --mode DP --client 5 --epsilon 5 --expname DP_data0_e5 --train_data train_0 --test_data test_0

    parser = argparse.ArgumentParser(description='dfsa')
    parser.add_argument('--device', default='cuda:0', help='default: cuda:0')
    parser.add_argument('--epochs', type=int, default=50, help='default: 50')
    parser.add_argument('--batch_size', type=int, default=1, help='default: 1')
    parser.add_argument('--lr', type=float, default=0.001, help='default: 0.01')
    parser.add_argument('--epsilon', type=int, default=100, help='default: 100')
    parser.add_argument('--delta', type=int, default=10e-5, help='default: 0.5')
    parser.add_argument('--mode', default='SGD', help='default: SGD, {SGD, SIGNSGD, DP, DPSIGNSGD}')
    parser.add_argument('--client', type=int, default=3, help='default: 3')
    parser.add_argument('--l2_clip', type=int, default=5, help='default: 5')
    parser.add_argument('--nprocess', type=int, default=100, help='default: 20')
    parser.add_argument('--dataset', default="yan",help='dataset name')
    parser.add_argument('--expname', help='experiment name')
    parser.add_argument('--train_data', default='train', help='will load data/{}.npy')
    parser.add_argument('--test_data', default='test', help='will load data/{}.npy')
    parser.add_argument('--shuffle_model', type=int,default=0, help='0: off, 1: on')
    args = parser.parse_args()

    dataset_name = args.dataset
    device_name = args.device
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    lr = args.lr
    epsilon = args.epsilon
    delta = args.delta
    mode = args.mode
    numberClients = args.client
    model_path = 'model'
    l2_norm_clip = args.l2_clip
    expname = args.expname
    train_index=args.train_data
    test_index=args.test_data
    nprocess=args.nprocess
    if args.shuffle_model==0:
        shuffle_model=False
    elif args.shuffle_model==1:
        shuffle_model=True


    logger = log_creater(output_dir=os.path.join(root, "output","SC",
                                                 "{}_epsilon_{}".format(expname,epsilon)),
                         expname=expname+"_"+dataset_name +"_{}".format(epsilon))

    logger.info("Mode: {}".format(mode))
    logger.info("Epochs: {}".format(EPOCHS))
    logger.info("lr: {}".format(lr))
    logger.info("batch size: {}".format(BATCH_SIZE))
    logger.info("epsilon: {}".format(epsilon))
    logger.info("delta: {}".format(delta))

    device = torch.device(device_name)

    trainDatasets = getSingleCellDataset(numberClients,dataset_name)

    trainLoaders=[DataLoader(dataset, BATCH_SIZE, True) for dataset in trainDatasets]


    print("k = {}".format(k))
    print("input_size = {}".format(input_size))
    # Loss Function
    criterion = nn.MSELoss()

    # Model
    serverModel = Model(input_size=input_size,
                        output_size=input_size)
    serverModel = serverModel.to(device)
    serverModel = torch.nn.DataParallel(serverModel).cuda()

    # start train
    trained_model , output = Train(logger,
                                   dataset_name,
                                    trainLoaders,
                                    serverModel,
                                    criterion,
                                    device=device,
                                    num_epochs=EPOCHS,
                                    model_path=model_path,
                                    mode=mode)


    output_file = os.path.join(root,
                               "output", "SC",
                               "{}_epsilon_{}".format(expname,epsilon),
                               "{}_{}.npy".format(expname,dataset_name))
    if not os.path.exists(os.path.dirname(output_file)):
        os.mkdir(os.path.dirname(output_file))

    output["label_dic"] = label_dic

    np.save(output_file, output)
