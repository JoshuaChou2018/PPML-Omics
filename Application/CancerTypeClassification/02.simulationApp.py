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
import argparse
from multiprocessing import Pool
import random
from sklearn.metrics import f1_score

root = '.'

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


def save_checkpoint(model, is_best, checkpoint,log, epoch):
    global mode
    global expname

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

        self.hidden1 = nn.Linear(input_size,2048) # 2048
        self.hidden2 = nn.Linear(2048, 1024)  # hidden layer
        self.hidden3 = nn.Linear(1024, 256)
        self.hidden4 = nn.Linear(256, output_size)  # output layer

    def forward(self, x):
        x = F.relu(self.hidden1(x))  # activation function for hidden layer
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = self.hidden4(x)  # linear output
        x = F.log_softmax(x,dim=1)
        return x

def cancerType(type):
    global cancerTypes
    idx=cancerTypes.index(type)
    #out=np.zeros(len(cancerTypes))
    #out[idx]=1
    #return out
    return idx

class Clinical_Data(Dataset):
    def __init__(self, index):
        print("[INFO] Loading {}".format(os.path.join(root,"data/{}.npy".format(index))))
        self.data = np.load(os.path.join(root,"data/{}.npy".format(index)),allow_pickle=True).item()
        # self.samples dic
        # 'TCGA-OR-A5J1-01A-11R-A29S-07':
        self.samples = self.data['samples']
        # feature_dic : 'ABCA7|10347',etc
        self.feature_dic = self.data['features']

        print("[INFO] {} Data has {} samples".format(index, len(self.samples)))

        # print("=> Preprocessing Data")
        # for patient_id in tqdm(self.samples.keys()):
        #     information = {
        #         "id":       patient_id,
        #         "genes":    [float(i) for i in self.samples[patient_id]["genes"]],
        #         "day2birth":float(self.samples[patient_id]["day2birth"]),
        #         "day2death": float(self.samples[patient_id]["day2death"]),
        #         "dayrecord": float(self.samples[patient_id]["dayrecord"]),
        #         "cancertype": self.samples[patient_id]["cancertype"]
        #     }
        #     self.data.append(information)

    def __getitem__(self, idx):
        data_dic = copy.deepcopy(self.samples[idx])
        genes = data_dic['genes']
        #label = data_dic['cancertype']
        label=cancerType(data_dic['cancertype'])

        #label = data_dic['day2death']

        del data_dic['genes']
        info = data_dic

        # Genes: 20531 len Tensor
        # lable: float value
        # info:  dic info
        return torch.Tensor(genes), label, info

    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small batch size of sets.
        return len(self.samples)

    def __alltypes__(self):
        out=[]
        for idx in range(len(self.samples)):
            out.append(self.samples[idx]['cancertype'])
        return out

def initialClientModel(serverModel,device):
    global cancerTypes
    clientModel=Model(input_size=20531,output_size=len(cancerTypes))
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

def accuracy(preds, labels):
    acc=(torch.tensor(preds).argmax(dim=1) == torch.tensor(labels).squeeze()).sum()/len(labels)
    return acc

def f1(preds, labels):
    y_pred=torch.tensor(preds).argmax(dim=1)
    y_true=torch.tensor(labels).squeeze()
    macro_f1=f1_score(y_true, y_pred, average='macro')
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    return macro_f1,micro_f1


def Train(logger,
          trainLoaders,
          testLoader,
          serverModel,
          criterions,
          device,
          num_epochs=25,
          model_path="model",
          mode='SGD'):

    global lr
    global epsilon
    global delta
    global numberClients
    global l2_norm_clip
    global shuffle_model

    loss_avg = RunningAverage()

    best_loss = 10 ** 15
    best_acc = 0
    is_best = False

    #optimizer = torch.optim.Adam(serverModel.parameters(), lr=lr)

    for epoch in range(num_epochs):

        logger.info('Epoch {}/{}'.format(epoch+1, num_epochs))
        labels = []
        preds = []
        #serverModel.train()

        if True:

            client_Model_list = []

            for client_idx in range(numberClients):
                logger.info('client id {}'.format(client_idx))
                trainLoader = trainLoaders[client_idx]

                # initial clientModel
                clientModel = initialClientModel(serverModel, device=device)
                if mode == 'SGD':
                    optimizer = torch.optim.SGD(clientModel.parameters(), lr=lr)
                else:
                    torch.nn.utils.clip_grad_norm_(clientModel.parameters(), l2_norm_clip,norm_type=2)
                    optimizer = torch.optim.SGD(clientModel.parameters(), lr=lr)
                    # optimizer = DPSGD(l2_norm_clip=l2_norm_clip,params=clientModel.parameters(),lr=lr)
                clientModel.train()

                with tqdm(total=len(trainLoader)) as t:
                    for genes, label, info in tqdm(trainLoader):
                        genes, label = genes.to(device), label.to(device, dtype=torch.long)
                        # serverModel.zero_grad()
                        # pred = serverModel(genes)
                        clientModel.zero_grad()
                        pred = clientModel(genes)

                        loss = criterions(pred, label)
                        loss.backward()
                        optimizer.step()
                        loss_avg.update(loss.item())
                        # statistics

                        labels += list(label.detach().cpu().numpy())
                        preds += list(pred.detach().cpu().numpy())

                        t.set_postfix(
                            loss_avg='{:05.3f}'.format(loss_avg()),
                            total_loss='{:05.3f}'.format(loss.item()),
                        )
                        t.update()
                client_Model_list.append(clientModel)
                logger.info("Length of model list: {}".format(len(client_Model_list)))

            # Shuffle the client_Model_list
            if shuffle_model == True:
                logger.info("=> Shuffling the client model")
                random.shuffle(client_Model_list)

            for clientModel in client_Model_list:
                grads = compute_grad_update(serverModel, clientModel, lr, device)
                # update to serverModel
                serverModel = updateServerModel(serverModel, grads, mode=mode, lr=lr, device=device, epsilon=epsilon,
                                                delta=delta)

        acc=accuracy(preds, labels)
        macro_f1, micro_f1 = f1(preds, labels)
        logger.info('Epoch Training loss: {}, acc: {}, macro_f1: {}, micro_f1: {}'.format(loss_avg(), acc,macro_f1,micro_f1))

        serverModel.eval()
        test_loss, test_acc = Test(logger,testLoader,serverModel,criterions,device)

        if  test_acc > best_acc:
            is_best = True
            best_loss = test_loss
            best_acc = test_acc
        else:
            is_best = False

        save_checkpoint(serverModel, is_best, model_path, logger, epoch)

    return serverModel


def Test(logger,
         test_loader,
         model,
         criterions,
         device
         ):
    loss_avg = RunningAverage()

    logger.info("Testing...")

    labels = []
    preds = []

    with tqdm(total=len(test_loader)) as t:
        for genes, label, info in tqdm(test_loader):
            genes, label = genes.to(device), label.to(device)

            pred = model(genes)
            loss = criterions(pred, label)
            loss_avg.update(loss.item())
            # statistics

            labels += list(label.detach().cpu().numpy())
            preds += list(pred.detach().cpu().numpy())

            t.set_postfix(loss_avg='{:05.3f}'.format(loss_avg()),
                              )
            t.update()

    acc = accuracy(preds, labels)
    macro_f1, micro_f1=f1(preds, labels)
    logger.info('Test loss: {} Test, acc: {}, macro_f1: {}, micro_f1: {}'.format(loss_avg(),acc,macro_f1,micro_f1))
    return loss_avg(), acc

def getSurvivalDataset(numClients, train_index='train', test_index='test'):

    global logger

    trainDataset = Clinical_Data(index=train_index)
    testDataset = Clinical_Data(index=test_index)

    train_counts={}
    test_counts={}
    for t in cancerTypes:
        train_counts[t]=0
        test_counts[t]=0

    for t in trainDataset.__alltypes__():
        train_counts[t]+=1
    for t in testDataset.__alltypes__():
        test_counts[t]+=1

    logger.info('Train Counts: {}'.format(train_counts))
    logger.info('Test Counts: {}'.format(test_counts))

    numItems = np.int(np.floor(len(trainDataset) / numClients))
    trainDatasets=[]
    if numClients!=1:
        for dataset in torch.utils.data.random_split(trainDataset, [numItems for i in range(numClients)]+[len(trainDataset)-numItems*numClients]):
            trainDatasets.append(dataset)
    else:
        trainDatasets.append(trainDataset)

    return trainDatasets, testDataset

if __name__ == '__main__':

    # python FLDP_CC_simulation_App_H3_R1_D0.py --mode DP --client 5 --epsilon 5 --expname DP_data0_e5 --train_data train_0 --test_data test_0

    parser = argparse.ArgumentParser(description='dfsa')
    parser.add_argument('--device', default='cuda:0', help='default: cuda:0')
    parser.add_argument('--epochs', type=int, default=10, help='default: 10')
    parser.add_argument('--batch_size', type=int, default=1, help='default: 1')
    parser.add_argument('--lr', type=float, default=0.001, help='default: 0.01')
    parser.add_argument('--epsilon', type=float, default=1, help='default: 1')
    parser.add_argument('--delta', type=float, default=10e-5, help='default: 10e-5')
    parser.add_argument('--mode', default='SGD', help='default: SGD, {SGD, SIGNSGD, DP, DPSIGNSGD}')
    parser.add_argument('--client', type=int, default=3, help='default: 3')
    parser.add_argument('--l2_clip', type=int, default=5, help='default: 5')
    parser.add_argument('--nprocess', type=int, default=100, help='default: 20')
    parser.add_argument('--expname', help='experiment name')
    parser.add_argument('--train_data', default='train', help='will load data/{}.npy')
    parser.add_argument('--test_data', default='test', help='will load data/{}.npy')
    parser.add_argument('--shuffle_model', type=int,default=0, help='0: off, 1: on')
    args = parser.parse_args()

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

    if 'cuda' in device_name:
        os.environ['CUDA_VISIBLE_DEVICES'] = device_name.split(':')[1]
        device_name = 'cuda:0'

    cancerTypes = ['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'COADREAD', 'DLBC', 'ESCA', 'GBM', 'GBMLGG', 'HNSC',
                   'KICH', 'KIPAN', 'KIRC', 'KIRP', 'LAML', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'MESO', 'OV', 'PAAD', 'PCPG',
                   'PRAD', 'READ', 'SARC', 'SKCM', 'STAD', 'STES', 'TGCT', 'THCA', 'THYM', 'UCEC', 'UCS', 'UVM']

    logger = log_creater(output_dir='log', expname=expname)

    logger.info("Mode: {}".format(mode))
    logger.info("Epochs: {}".format(EPOCHS))
    logger.info("lr: {}".format(lr))
    logger.info("batch size: {}".format(BATCH_SIZE))
    logger.info("epsilon: {}".format(epsilon))
    logger.info("delta: {}".format(delta))

    device = torch.device(device_name)

    trainDatasets, testDataset = getSurvivalDataset(numberClients,train_index=train_index, test_index=test_index)
    print(trainDatasets)
    trainLoaders=[DataLoader(dataset, BATCH_SIZE, True) for dataset in trainDatasets]
    testLoader = DataLoader(testDataset, BATCH_SIZE, False)

    # Loss Function
    criterion = nn.CrossEntropyLoss()

    # Model
    serverModel = Model(input_size=20531,output_size=len(cancerTypes))
    serverModel = serverModel.to(device)
    serverModel = torch.nn.DataParallel(serverModel).cuda()

    # start train
    trained_model = Train(logger,
                        trainLoaders,
                        testLoader,
                        serverModel,
                        criterions=criterion,
                        device=device,
                        num_epochs=EPOCHS,
                        model_path=model_path,
                        mode=mode)

    Test(logger,
         testLoader,
         model=trained_model,
         criterions=criterion,
         device=device)
