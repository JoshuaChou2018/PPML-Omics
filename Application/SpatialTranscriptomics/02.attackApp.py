import time
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, transforms

#############
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
import torch.nn.functional as F
import os
import numpy as np
from PIL.Image import open as read_image
from torch.utils.data import DataLoader, Dataset
import torch
import PIL
import torchvision
import glob
import openslide
import pickle
import pathlib
import collections
import pandas
import argparse
import traceback
import socket
import random
import time
import sklearn
import sklearn.ensemble
import logging

raw_data_path = './data/hist2tscript'
processed_data_path = './data/hist2tscript-patch'

def get_spatial_patients():
    """
    Returns a dict of patients to sections.

    The keys of the dict are patient names (str), and the values are lists of
    section names (str).
    """
    images = glob.glob(raw_data_path + r'/*.jpg')

    patient_section = map(lambda x: x.split("/")[-1][:-4].split("_"), images)
    patient = collections.defaultdict(list)
    for (a, p, s) in patient_section:
        patient[p].append(s)
    return patient

def get_finetune_parameters(model, layers, randomize):
    if layers is None:
        return model.parameters()
    else:
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        # if isinstance(torchvision.models.vgg.VGG):
        #     parameters = model.classifier[-4:].parameters()
        # elif isinstance(torchvision.models.densenet.DenseNet):
        #     parameters = model.classifier.parameters()
        # elif (isinstance(torchvision, torchvision.models.resnet.ResNet) or
        #       isinstance(torchvision, torchvision.models.inception.Inception3)):
        #     parameters = model.fc.parameters()
        if isinstance(model, torchvision.models.densenet.DenseNet):
            modules = [model.classifier]
            modules.extend(model.features[-(layers - 1):])
        else:
            raise NotImplementedError()

        parameters = []
        for module in modules:
            if randomize:
                for m in module.modules():
                    if isinstance(m, torch.nn.Conv2d):
                        torch.nn.init.kaiming_normal_(m.weight)
                    elif isinstance(m, torch.nn.BatchNorm2d):
                        torch.nn.init.constant_(m.weight, 1)
                        torch.nn.init.constant_(m.bias, 0)
                    elif isinstance(m, torch.nn.Linear):
                        m.reset_parameters()
                        torch.nn.init.constant_(m.bias, 0)

            parameters += list(module.parameters())
        return parameters

def set_out_features(model, outputs):
    """Changes number of outputs for the model.

    The change occurs in-place, but the new model is also returned."""

    if (isinstance(model, torchvision.models.AlexNet) or
        isinstance(model, torchvision.models.VGG)):
        inputs = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(inputs, outputs, bias=True)
        model.classifier[-1].weight.data.zero_()
        model.classifier[-1].bias.data.zero_()
    elif (isinstance(model, torchvision.models.ResNet) or
          isinstance(model, torchvision.models.Inception3)):
        inputs = model.fc.in_features
        model.fc = torch.nn.Linear(inputs, outputs, bias=True)
        model.fc.weight.data.zero_()
        model.fc.bias.data.zero_()
    elif isinstance(model, torchvision.models.DenseNet):
        inputs = model.classifier.in_features
        model.classifier = torch.nn.Linear(inputs, outputs, bias=True)
        model.classifier.weight.data.zero_()
        model.classifier.bias.data.zero_()
    else:
        raise NotImplementedError()

    return model

# We used model VGG16
class IdentityDict(dict):
    """This variant of a dict defaults to the identity function if a key has
    no corresponding value.

    https://stackoverflow.com/questions/6229073/how-to-make-a-python-dictionary-that-returns-key-for-keys-missing-from-the-dicti
    """
    def __missing__(self, key):
        return key


class Spatial(torch.utils.data.Dataset):

    def __init__(self,
                 patient=None,
                 transform=None,
                 window=224,
                 cache=False,
                 raw_root = raw_data_path,
                 root=processed_data_path,
                 gene_filter="tumor",
                 load_image=True,
                 gene_transform="log",
                 downsample=1,
                 norm=None,
                 feature=False):

        self.dataset = sorted(glob.glob("{}/*/*/*.npz".format(root)))
        if patient is not None:
            # Can specify (patient, section) or only patient (take all sections)
            self.dataset = [d for d in self.dataset if ((d.split("/")[-2] in patient) or ((d.split("/")[-2], d.split("/")[-1].split("_")[0]) in patient))]

            # TODO: if patient == [], then many things downstream bug out
            # how to handle this case?
            # Could just throw an error

        # TODO: filter things that are too close to edge?

        self.transform = transform
        self.window = window
        self.downsample = downsample
        self.cache = cache
        self.root = root
        self.load_image = load_image
        self.gene_transform = gene_transform
        self.norm = norm
        self.feature = feature

        with open(root + "/subtype.pkl", "rb") as f:
            self.subtype = pickle.load(f)
        with open(root + "/gene.pkl", "rb") as f:
            self.ensg_names = pickle.load(f)
        try:
            with open(processed_data_path + "/ensembl.pkl", "rb") as f:
                symbol = pickle.load(f)
        except FileNotFoundError:
            ensembl = pandas.read_table(processed_data_path + "/custom.txt", sep="\t")
            # TODO: should just return Ensembl ID if no name available
            symbol = IdentityDict()
            for (index, row) in ensembl.iterrows():
                symbol[row["Ensembl ID(supplied by Ensembl)"]] = row["Approved symbol"]
            with open(processed_data_path + "/ensembl.pkl", "wb") as f:
                pickle.dump(symbol, f)

        self.gene_names = list(map(lambda x: symbol[x], self.ensg_names))
        self.mean_expression = np.load(root + "/mean_expression.npy")
        self.median_expression = np.load(root + "/median_expression.npy")

        self.slide = collections.defaultdict(dict)
        # TODO: this can be parallelized
        for (patient, section) in set([(d.split("/")[-2], d.split("/")[-1].split("_")[0]) for d in self.dataset]):
            self.slide[patient][section] = openslide.open_slide(raw_root + '/HE_' +patient + '_' + section + '.tif')

        if gene_filter is None or gene_filter == "none":
            self.gene_filter = None
        elif gene_filter == "high":
            self.gene_filter = np.array([m > 1. for m in self.mean_expression])
        elif gene_filter == "tumor":
            # These are the 10 genes with the largest difference in expression between tumor and normal tissue
            # Printed by save_counts.py
            # self.tumor_genes = ["PABPC1", "GNAS", "HSP90AB1", "TFF3", "ATP1A1", "COX6C", "B2M", "FASN", "ACTG1", "HLA-B"]
            self.tumor_genes = ["FASN"]
            self.gene_filter = np.array([g in self.tumor_genes for g in self.gene_names])
        elif isinstance(gene_filter, list):
            self.gene_filter = np.array([g in gene_filter or e in gene_filter for (g, e) in zip(self.gene_names, self.ensg_names)])
        elif isinstance(gene_filter, int):
            keep = set(list(zip(*sorted(zip(self.mean_expression, range(self.mean_expression.shape[0])))[::-1][:gene_filter]))[1])
            self.gene_filter = np.array([i in keep for i in range(len(self.gene_names))])
        else:
            raise ValueError()

        if self.gene_filter is not None:
            self.ensg_names = [n for (n, f) in zip(self.ensg_names, self.gene_filter) if f]
            self.gene_names = [n for (n, f) in zip(self.gene_names, self.gene_filter) if f]
            self.mean_expression = self.mean_expression[self.gene_filter]
            self.median_expression = self.median_expression[self.gene_filter]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        npz = np.load(self.dataset[index])

        count   = npz["count"]
        tumor   = npz["tumor"]
        pixel   = npz["pixel"]
        patient = npz["patient"][0]
        section = npz["section"][0]
        coord   = npz["index"]

        feature_filename = os.path.splitext(self.dataset[index])[0] + "_feature.npy"
        need_image = self.feature and not os.path.isfile(feature_filename)

        if self.load_image or need_image:
            orig_window = self.window * self.downsample
            cached_image = "{}/{}/{}/{}_{}_{}_{}_{}_{}.tif".format(self.root, self.subtype[patient], patient, patient, section, orig_window, self.downsample, coord[0], coord[1])
            if self.cache and pathlib.Path(cached_image).exists():
                X = PIL.Image.open(cached_image)
            else:
                slide = self.slide[patient][section]
                X = slide.read_region((pixel[0] - orig_window // 2, pixel[1] - orig_window // 2), 0, (orig_window, orig_window))
                X = X.convert("RGB")

                if self.cache:
                    X.save(cached_image)

                # TODO: check downsample
                if self.downsample != 1:
                    X = torchvision.transforms.Resize((self.window, self.window))(X)

            if self.transform is not None:
                X = self.transform(X)

        # if self.feature:
        #     try:
        #         f = np.load(feature_filename)
        #     except FileNotFoundError:
        #         # f = stnet.util.histology.features(X).numpy()
        #         # np.save(feature_filename, f)
        #     f = torch.Tensor(f[0, :])
        # else:
        #     f = []
        f = []
        if not self.load_image:
            X = []

        Z = np.sum(count)
        n = count.shape[0]
        if self.gene_filter is not None:
            count = count[self.gene_filter]
        y = torch.as_tensor(count, dtype=torch.float)

        tumor = torch.as_tensor([1 if tumor else 0])
        coord = torch.as_tensor(coord)
        index = torch.as_tensor([index])

        if self.norm is None or self.norm == "none":
            if self.gene_transform == "log":
                y = torch.log(1 + y)
            elif self.gene_transform == "sqrt":
                y = torch.sqrt(y)
            else:
                assert(self.gene_transform is None or self.gene_transform == "none")
        elif self.norm == "norm":
            if self.gene_transform == "log":
                y = torch.log((1 + y) / (n + Z))
            else:
                assert(self.gene_transform is None or self.gene_transform == "none")
                y = y / Z
        elif self.norm == "normfilter":
            if self.gene_transform == "log":
                y = torch.log((1 + y) / torch.sum(1 + y))
            else:
                assert(self.gene_transform is None or self.gene_transform == "none")
                y = y / torch.sum(y)
        elif self.norm == "normpat":
            if self.gene_transform == "log":
                y = torch.log((1 + y) / self.p_median[patient])
            else:
                assert(self.gene_transform is None or self.gene_transform == "none")
                y = y / self.ps_median[(patient, section)]
        elif self.norm == "normsec":
            if self.gene_transform == "log":
                y = torch.log((1 + y) / self.ps_median[(patient, section)])
            else:
                assert(self.gene_transform is None or self.gene_transform == "none")
                y = y / self.ps_median[(patient, section)]
        else:
            raise ValueError()

        ### X: [batchsize, channel ,window, window]
        ### y:
        return X, tumor, y, coord, index, patient, section, pixel, f

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

def initialClientModel(serverModel,device):
    global cancerTypes
    clientModel=get_model(trainDatasets[0])
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
    #logger.info('sigma: {}'.format(sigma))
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

def get_model(train_dataset):

        ### Model setup ###
        # if args.model == "rf":
        #     pass
        # elif args.model == "inception_v3":
        #     model = torchvision.models.__dict__[args.model](pretrained=args.pretrained, aux_logits=False)
        # else:
        #     model = torchvision.models.__dict__[args.model](pretrained=args.pretrained)

        model = torchvision.models.__dict__[args.model](pretrained=args.pretrained)

        start_epoch = 0
        if args.model != "linear" and args.model != "rf":
            # Replace last layer
            # TODO: if loading weights, should just match outs
            set_out_features(model, outputs)

            if args.gpu:
                model = torch.nn.DataParallel(model)
            model.to(device)


            ### Optimizer setup ###
            parameters = get_finetune_parameters(model,None,None)
            optim = torch.optim.__dict__['SGD'](parameters, lr=1e-5, momentum=0.9, weight_decay=0)

            if args.load is not None:
                model.load_state_dict(torch.load(args.load)["model"])

            ### Reload parameters from incomplete run ###
            # if args.restart:
            #     for epoch in range(args.epochs)[::-1]:
            #         if os.path.isfile(args.checkpoint + str(epoch + 1) + ".pt"):
            #             start_epoch = epoch + 1
            #             try:
            #                 checkpoint = torch.load(args.checkpoint + str(epoch + 1) + ".pt")
            #             except:
            #                 # A runtime error is thrown if the checkpoint is corrupted (most likely killed while saving)
            #                 # Continue checking for earlier checkpoints
            #                 logger.info("Detected corrupted checkpoint at epoch #{}. Skipping checkpoint.".format(start_epoch))
            #                 continue
            #             model.load_state_dict(checkpoint["model"])
            #             optim.load_state_dict(checkpoint["optim"])
            #             logger.info("Detected run stopped at epoch #{}. Restarting from checkpoint.".format(start_epoch))
            #             break

        global mean_expression
        global mean_expression_normal
        global mean_expression_tumor

        if args.model != "rf":
            m = model
            if args.gpu:
                m = m.module

            if (isinstance(m, torchvision.models.AlexNet) or
                    isinstance(m, torchvision.models.VGG)):
                last = m.classifier[-1]
            elif isinstance(m, torchvision.models.DenseNet):
                last = m.classifier
            elif (isinstance(m, torchvision.models.ResNet) or
                  isinstance(m, torchvision.models.Inception3)):
                last = m.fc
            else:
                raise NotImplementedError()

            if args.load is None and start_epoch == 0:
                last.weight.data.zero_()
                if args.task == "gene":
                    last.bias.data = mean_expression.clone()
                elif args.task == "geneb":
                    last.bias.data.zero_()
                elif args.task == "count":
                    mean_expression = torch.sum(mean_expression, 0, keepdim=True)
                    mean_expression_tumor = torch.sum(mean_expression_tumor, 0, keepdim=True)
                    mean_expression_normal = torch.sum(mean_expression_normal, 0, keepdim=True)
                    last.bias.data = mean_expression.clone()
                else:
                    raise ValueError()

            if args.gene_mask is not None:
                args.gene_mask = torch.Tensor([args.gene_mask])
                args.gene_mask = args.gene_mask.to(device)

        return model

def compute_mean_expression():

    # Compute mean expression for initial params in model and as baseline
    train_dataset=fulltrainDataset
    train_loader=fulltrainLoader
    start_epoch = 0
    if args.task == "gene" or args.task == "geneb" or args.task == "count":
        t = time.time()

        mean_expression = torch.zeros(train_dataset[0][2].shape)
        mean_expression_tumor = torch.zeros(train_dataset[0][2].shape)
        mean_expression_normal = torch.zeros(train_dataset[0][2].shape)
        tumor = 0
        normal = 0
        load_image = train_loader.dataset.load_image
        train_loader.dataset.load_image = False  # Temporarily stop loading images to save time
        for (i, (_, y, gene, *_)) in enumerate(train_loader):
            print("{:8d} / {:d}:    {:4.0f} / {:4.0f} seconds".format(i + 1, len(train_loader), time.time() - t,
                                                                      (time.time() - t) * len(train_loader) / (i + 1)),
                  end="\r", flush=True)
            mean_expression += torch.sum(gene, 0)
            mean_expression_tumor += torch.sum(y.float() * gene, 0)
            mean_expression_normal += torch.sum((1 - y).float() * gene, 0)
            tumor += torch.sum(y).detach().numpy()
            normal += torch.sum(1 - y).detach().numpy()
        train_loader.dataset.load_image = load_image

        mean_expression /= float(tumor + normal)
        mean_expression_tumor /= float(tumor)
        mean_expression_normal /= float(normal)
        median_expression = torch.log(1 + torch.Tensor(train_dataset.median_expression))

        mean_expression = mean_expression.to(device)
        mean_expression_tumor = mean_expression_tumor.to(device)
        mean_expression_normal = mean_expression_normal.to(device)
        median_expression = median_expression.to(device)
        logger.info("Computing mean expression took {}".format(time.time() - t))

    return mean_expression, mean_expression_tumor, mean_expression_normal, median_expression

def Train_ST(train_dataset,train_loader,test_dataset,test_loader,model,optim,epoch,dataset='train'):

        # Find number of required outputs
        if args.task == "tumor":
            outputs = 2
        elif args.task == "gene":
            outputs = train_dataset[0][2].shape[0]
        elif args.task == "geneb":
            outputs = 2 * train_dataset[0][2].shape[0]
        elif args.task == "count":
            outputs = 1

        ### Training Loop ###
        if True:
            logger.info("Epoch #" + str(epoch + 1))
            if True:
                if dataset=='train':
                    loader=train_loader
                elif dataset=='test':
                    loader=test_loader
                t = time.time()
                torch.set_grad_enabled(dataset == "train")
                if args.model != "rf":
                    model.train(dataset == "train")
                total = 0
                total_mean = 0
                total_type = 0
                correct = 0
                positive = 0
                mse = np.zeros(train_dataset[0][2].shape)
                mse_mean = np.zeros(train_dataset[0][2].shape)
                mse_type = np.zeros(train_dataset[0][2].shape)
                features = []
                genes = []
                predictions = []
                counts = []
                tumor = []
                coord = []
                patient = []
                section = []
                pixel = []

                save_pred = (dataset == "test" and args.pred_root is not None and (args.save_pred_every is None or (epoch + 1) % args.save_pred_every == 0))

                n = 0
                logger.info(dataset + ":")
                with tqdm(total=len(loader)) as tq:
                    i=0
                    for X, y, gene, c, ind, pat, s, pix, f in tqdm(loader):
                        if save_pred:
                            counts.append(gene.detach().numpy())
                            tumor.append(y.detach().numpy())
                            coord.append(c.detach().numpy())
                            patient += pat
                            section += s
                            pixel.append(pix.detach().numpy())

                        X = X.to(device)
                        y = y.to(device)
                        gene = gene.to(device)

                        if dataset == "test" and args.average:
                            batch, n_sym, c, h, w = X.shape
                            X = X.view(-1, c, h, w)
                        if args.model == "rf":
                            if dataset == "train":
                                features.append(f.detach().numpy())
                                genes.append(gene.cpu().detach().numpy())
                                pred = gene
                            else:
                                pred = torch.Tensor(model.predict(f.detach().numpy())).to(device)
                        else:
                            pred = model(X)
                        if dataset == "test" and args.average:
                            pred = pred.view(batch, n_sym, -1).mean(1)
                        if save_pred:
                            predictions.append(pred.cpu().detach().numpy())

                        if args.task == "tumor":
                            y = torch.squeeze(y, dim=1)
                            loss = torch.nn.functional.cross_entropy(pred, y, reduction='sum')
                            correct += torch.sum(torch.argmax(pred, dim=1) == y).cpu().detach().numpy()
                            positive += torch.sum(y).cpu().detach().numpy()
                        elif args.task == "gene":
                            if args.gene_mask is None:
                                loss = torch.sum((pred - gene) ** 2) / outputs
                            else:
                                loss = torch.sum(args.gene_mask * (pred - gene) ** 2) / torch.sum(args.gene_mask)
                            mse += torch.sum((pred - gene) ** 2, 0).cpu().detach().numpy()

                            # Evaluating baseline performance
                            total_mean += (torch.sum((mean_expression - gene) ** 2) / outputs).cpu().detach().numpy()
                            mse_mean   += torch.sum((mean_expression - gene) ** 2, 0).cpu().detach().numpy()
                            y = y.float()
                            total_type += (torch.sum((y * mean_expression_tumor + (1 - y) * mean_expression_normal - gene) ** 2) / outputs).cpu().detach().numpy()
                            mse_type += torch.sum((y * mean_expression_tumor + (1 - y) * mean_expression_normal - gene) ** 2, 0).cpu().detach().numpy()
                        elif args.task == "geneb":
                            gene = (gene > torch.log(1 + median_expression)).type(torch.int64)
                            # TODO: gene mask needs to work here too
                            loss = torch.nn.functional.cross_entropy(pred.reshape(-1, 2), gene.reshape(-1), reduction='sum') / outputs
                        elif args.task == "count":
                            gene = torch.sum(gene, 1, keepdim=True)
                            loss = torch.sum((pred - gene) ** 2)
                            total_mean += torch.sum((mean_expression - gene) ** 2).cpu().detach().numpy()
                            total_type += torch.sum((y * mean_expression_tumor + (1 - y) * mean_expression_normal - gene) ** 2).cpu().detach().numpy()
                        else:
                            raise ValueError()
                        #print(loss.cpu().detach().numpy())
                        total += loss.cpu().detach().numpy()
                        n += y.shape[0]

                        message = ""
                        message += "{:8d} / {:d} ({:4.0f} / {:4.0f}):".format(i + 1, len(loader), time.time() - t, (time.time() - t) * len(loader) / (i + 1))
                        i=i+1
                        message += "    Loss={:.3f}".format(total / n)
                        if args.task == "tumor":
                            message += "    Accuracy={:.3f}    Tumor={:.3f}".format(correct / n, positive / n)
                        if args.task == "gene":
                            message += "    MSE={:.3f}    Type:{:.3f}".format(total_mean / n, total_type / n)
                        #logger.info(message)
                        #logger.debug(message)

                        if dataset == "train" and (args.model != "rf"):
                            optim.zero_grad()
                            loss.backward()
                            optim.step()
                logger.info("    Loss:       " + str(total / len(loader.dataset)))
                if args.task == "tumor":
                    logger.info("    Accuracy:   " + str(correct / len(loader.dataset)))
                    logger.info("    Percentage: " + str(positive / len(loader.dataset)))
                if args.task == "gene":
                    MSE=total / len(loader.dataset)
                    logger.info("    MSE:        " + str(total_mean / len(loader.dataset)))
                    logger.info("    Type:       " + str(total_type / len(loader.dataset)))
                    logger.info("    Best:       " + str(max((mse_mean - mse) / mse_mean)))
                    logger.info("    Worst:      " + str(min((mse_mean - mse) / mse_mean)))
                # TODO: debug messages for geneb and count are incomplete (also in the progress bar)

                if dataset == "train" and args.model == "rf":
                    features = np.concatenate(features)
                    genes = np.concatenate(genes)
                    if args.model == "linear":
                        model = sklearn.linear_model.LinearRegression().fit(features, genes)
                    else:
                        model = sklearn.ensemble.RandomForestRegressor(n_estimators=100).fit(features, genes)

                if save_pred:
                    predictions = np.concatenate(predictions)
                    counts = np.concatenate(counts)
                    tumor = np.concatenate(tumor)
                    coord = np.concatenate(coord)
                    pixel = np.concatenate(pixel)

                    if args.task == "tumor":
                        me = None
                        me_tumor = None
                        me_normal = None
                    else:
                        me = mean_expression.cpu().numpy(),
                        me_tumor = mean_expression_tumor.cpu().numpy(),
                        me_normal = mean_expression_normal.cpu().numpy(),

                    pathlib.Path(os.path.dirname(args.pred_root)).mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(args.pred_root + '/'+ str(epoch + 1),
                                        task=args.task,
                                        tumor=tumor,
                                        counts=counts,
                                        predictions=predictions,
                                        coord=coord,
                                        patient=patient,
                                        section=section,
                                        pixel=pixel,
                                        mean_expression=me,
                                        mean_expression_tumor=me_tumor,
                                        mean_expression_normal=me_normal,
                                        ensg_names=test_dataset.ensg_names,
                                        gene_names=test_dataset.gene_names,
                    )

                # Saving after test so that if information from test is needed, they will not get skipped
                if dataset == "test" and args.checkpoint is not None and ((epoch + 1) % args.checkpoint_every) == 0 and args.model != "rf":
                    pathlib.Path(os.path.dirname(args.checkpoint)).mkdir(parents=True, exist_ok=True)
                    # TODO: if model is on gpu, does loading automatically put on gpu?
                    # https://discuss.pytorch.org/t/how-to-store-model-which-trained-by-multi-gpus-dataparallel/6526
                    # torch.save(model.state_dict(), args.checkpoint + str(epoch + 1) + ".pt")
                    torch.save({
                        'model': model.state_dict(),
                        'optim' : optim.state_dict(),
                    }, args.checkpoint + '/'+ str(epoch + 1) + ".pt")

                    if epoch != 0 and (args.keep_checkpoints is None or (epoch + 1 - args.checkpoint_every) not in args.keep_checkpoints):
                        try:
                            os.remove(args.checkpoint+'/' + str(epoch + 1 - args.checkpoint_every) + ".pt")
                        except:
                            pass
        if dataset=='train':
            return model
        elif dataset=='test':
            return MSE


def Train(logger,
          trainLoaders,
          trainDatasets,
          testLoader,
          testDataset,
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

    best_MSE = np.inf
    is_best = False

    #optimizer = torch.optim.Adam(serverModel.parameters(), lr=lr)

    for epoch in range(num_epochs):

        logger.info('Epoch {}/{}'.format(epoch+1, num_epochs))
        labels = []
        preds = []
        serverModel.train()

        if shuffle_model==False:

            for client_idx in range(numberClients):
                logger.info('client id {}'.format(client_idx))
                trainLoader=trainLoaders[client_idx]
                train_dataset=trainDatasets[client_idx]

                # initial clientModel
                clientModel = initialClientModel(serverModel, device=device)
                if mode=='SGD':
                    optimizer = torch.optim.SGD(clientModel.parameters(), lr=lr)
                else:
                    torch.nn.utils.clip_grad_norm_(clientModel.parameters(), l2_norm_clip)
                    optimizer = torch.optim.SGD(clientModel.parameters(), lr=lr)
                    #optimizer = DPSGD(l2_norm_clip=l2_norm_clip,params=clientModel.parameters(),lr=lr)
                clientModel.train()

                clientModel=Train_ST(train_dataset,trainLoader,testDataset,testLoader,clientModel,optimizer,epoch,dataset='train')

                #serverModel=clientModel
                # calculate grad update
                grads=compute_grad_update(serverModel, clientModel,lr,device)

                # update to serverModel
                serverModel = updateServerModel(serverModel, grads, mode=mode, lr=lr, device=device,epsilon=epsilon, delta=delta)

        elif shuffle_model==True:

            client_Model_list = []

            for client_idx in range(numberClients):
                logger.info('client id {}'.format(client_idx))
                trainLoader = trainLoaders[client_idx]
                train_dataset = trainDatasets[client_idx]

                # initial clientModel
                clientModel = initialClientModel(serverModel, device=device)
                if mode == 'SGD':
                    optimizer = torch.optim.SGD(clientModel.parameters(), lr=lr)
                else:
                    torch.nn.utils.clip_grad_norm_(clientModel.parameters(), l2_norm_clip)
                    optimizer = torch.optim.SGD(clientModel.parameters(), lr=lr)
                    # optimizer = DPSGD(l2_norm_clip=l2_norm_clip,params=clientModel.parameters(),lr=lr)
                clientModel.train()

                clientModel = Train_ST(train_dataset, trainLoader, testDataset, testLoader, clientModel, optimizer,
                                       epoch, dataset='train')

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

        serverModel.eval()
        test_MSE = Test(logger,testLoader,testDataset,serverModel,criterions,device,optimizer,epoch)

        #"""
        if  test_MSE < best_MSE:
            is_best = True
            best_MSE = test_MSE
        else:
            is_best = False
        #"""
        #is_best=True

        save_checkpoint(serverModel, is_best, model_path, logger, epoch)

    return serverModel


def Test(logger,
         test_loader,
         testDataset,
         model,
         criterions,
         device,
         optim,
         epoch,
         ):
    loss_avg = RunningAverage()

    logger.info("Testing...")

    labels = []
    preds = []

    MSE = Train_ST(testDataset, test_loader, testDataset, testLoader, model, optim, epoch,
                           dataset='test')

    return MSE

def getSTDataset(numClients):

    global logger

    ### Seed RNGs ###
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

    ### generating mean_expression.npy ###
    logging.info("Computing statistics of dataset")
    gene = []
    for filename in tqdm(glob.glob("{}/*/*/*_*_*.npz".format(processed_data_path))):
        npz = np.load(filename)
        count = npz["count"]
        gene.append(np.expand_dims(count, 1))

    gene = np.concatenate(gene, 1)
    np.save(processed_data_path + "/mean_expression.npy", np.mean(gene, 1))
    np.save(processed_data_path + "/median_expression.npy", np.median(gene, 1))

    ### Split patients into folds ###
    patient = get_spatial_patients()

    train_patients = []
    test_patients = []
    n_test = round(args.test * len(patient))
    is_test = [True for i in range(n_test)] + [False for i in range(len(patient) - n_test)]
    random.shuffle(is_test)

    for (i, p) in enumerate(patient):
        # if args.trainpatients is None and args.testpatients is None:
        if is_test[i]:
            for s in patient[p]:
                test_patients.append((p, s))
        else:
            for s in patient[p]:
                train_patients.append((p, s))

    logger.info('Train Counts: {}'.format(len(train_patients)))
    logger.info('Test Counts: {}'.format(len(test_patients)))

    numItems = np.int(np.floor(len(train_patients) / numClients))
    trainDatasets=[]

    def train_patients2train_dataset(patients):
        train_dataset = Spatial(patients, window=args.window, gene_filter=args.gene_filter,
                                downsample=args.downsample, norm=args.norm, gene_transform=args.gene_transform,
                                transform=torchvision.transforms.ToTensor(), feature=(args.model == "rf"))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, num_workers=args.workers,
                                                   shuffle=True, pin_memory=args.gpu)

        # Estimate mean and covariance
        t = time.time()
        n_samples = 10
        mean = 0.
        std = 0.
        n = 0
        for (i, (X, *_)) in enumerate(train_loader):
            X = X.transpose(0, 1).contiguous().view(3, -1)
            n += X.shape[1]
            mean += torch.sum(X, dim=1)
            std += torch.sum(X ** 2, dim=1)
            if i > n_samples:
                break
        mean /= n
        std = torch.sqrt(std / n - mean ** 2)
        logger.info("Estimating mean (" + str(mean) + ") and std (" + str(std) + " took " + str(time.time() - t))

        transform = []

        transform.extend([torchvision.transforms.RandomHorizontalFlip(),
                          torchvision.transforms.RandomVerticalFlip(),
                          torchvision.transforms.RandomApply([torchvision.transforms.RandomRotation((90, 90))]),
                          torchvision.transforms.ToTensor(),
                          torchvision.transforms.Normalize(mean=mean, std=std)])
        transform = torchvision.transforms.Compose(transform)

        train_dataset.transform = transform
        return train_dataset

    def test_patients2test_dataset(patients, mean, std):
        #transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
        #                                            torchvision.transforms.Normalize(mean=mean, std=std)])
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        # TODO: random crops on test too?
        test_dataset = Spatial(test_patients, transform, window=args.window, gene_filter=args.gene_filter,
                               downsample=args.downsample, norm=args.norm, gene_transform=args.gene_transform,
                               feature=(args.model == "rf"))
        return test_dataset

    def getMeanStd(patients):
        train_dataset = Spatial(patients, window=args.window, gene_filter=args.gene_filter,
                                downsample=args.downsample, norm=args.norm, gene_transform=args.gene_transform,
                                transform=torchvision.transforms.ToTensor(), feature=(args.model == "rf"))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, num_workers=args.workers,
                                                   shuffle=True, pin_memory=args.gpu)

        # Estimate mean and covariance
        t = time.time()
        n_samples = 10
        mean = 0.
        std = 0.
        n = 0
        for (i, (X, *_)) in enumerate(train_loader):
            X = X.transpose(0, 1).contiguous().view(3, -1)
            n += X.shape[1]
            mean += torch.sum(X, dim=1)
            std += torch.sum(X ** 2, dim=1)
            if i > n_samples:
                break
        mean /= n
        std = torch.sqrt(std / n - mean ** 2)
        return mean,std

    if numClients!=1:
        if len(train_patients)-numItems*numClients!=0:
            for patients in torch.utils.data.random_split(train_patients, [numItems for i in range(numClients)]+[len(train_patients)-numItems*numClients]):
                trainDatasets.append(train_patients2train_dataset(patients))
        elif len(train_patients)-numItems*numClients==0:
            for patients in torch.utils.data.random_split(train_patients, [numItems for i in range(numClients)]):
                trainDatasets.append(train_patients2train_dataset(patients))
    else:
        trainDatasets.append(train_patients2train_dataset(train_patients))

    mean,std=getMeanStd(test_patients)
    testDataset=test_patients2test_dataset(test_patients,mean,std)
    fulltrainDataset=train_patients2train_dataset(train_patients)

    return trainDatasets, testDataset, fulltrainDataset
#############

def weights_init(m):
    try:
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.weight' % m._get_name())
    try:
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.bias' % m._get_name())

# input: 3x224x224
# output: 10

class LeNet(nn.Module):
    def __init__(self, channel=3, hideen=150528, num_classes=1):
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),
            act()
        )
        self.fc = nn.Sequential(
            nn.Linear(hideen, num_classes)
        )

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='dfsa')
    parser.add_argument('--device', default='cuda:0', help='default: cuda:0')
    parser.add_argument('--epochs', type=int, default=50, help='default: 50')
    parser.add_argument('--batch_size', type=int, default=1, help='default: 1')
    parser.add_argument('--lr', type=float, default=0.001, help='default: 0.01')
    parser.add_argument('--epsilon', type=float, default=100, help='default: 100')
    parser.add_argument('--delta', type=int, default=10e-5, help='default: 0.5')
    parser.add_argument('--mode', default='SGD', help='default: SGD, {SGD, SIGNSGD, DP, DPSIGNSGD}')
    parser.add_argument('--client', type=int, default=3, help='default: 3')
    parser.add_argument('--l2_clip', type=int, default=5, help='default: 5')
    parser.add_argument('--nprocess', type=int, default=100, help='default: 20')
    parser.add_argument('--expname', help='experiment name')
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
    nprocess=args.nprocess
    if args.shuffle_model==0:
        shuffle_model=False
    elif args.shuffle_model==1:
        shuffle_model=True

    args = argparse.ArgumentParser()
    args.window_raw = None
    args.test = 0.25
    args.window = 224
    args.gene_filter = 'tumor'
    args.downsample = 1
    args.gene_transform = 'log'
    args.model = 'vgg11'
    args.norm = None
    args.batch = BATCH_SIZE
    args.workers = nprocess
    args.gpu = True
    args.pretrained = True
    args.task = 'gene'
    args.load = None
    args.gene_mask = None
    args.epochs = EPOCHS

    args.trainpatients = None
    args.testpatients = None
    args.restart = None
    args.save_pred_every = None
    args.average = None
    args.checkpoint_every = 10
    args.keep_checkpoints = None

    if 'cuda' in device_name:
        os.environ['CUDA_VISIBLE_DEVICES'] = device_name.split(':')[1]
        device_name='cuda:0'

    """
    device_name = 'cuda:0'
    EPOCHS = 50
    BATCH_SIZE = 1
    lr = 0.01
    epsilon=100
    delta=0.5
    #mode='SGD'
    #mode='SIGNSGD'
    mode='DP'
    #mode='DPSIGNSGD'
    numberClients=3
    model_path='model'
    l2_norm_clip=5
    expname='{}_02'.format(mode)
    """

    logger = log_creater(output_dir='log', expname=expname)

    logger.info("Mode: {}".format(mode))
    logger.info("Epochs: {}".format(EPOCHS))
    logger.info("lr: {}".format(lr))
    logger.info("batch size: {}".format(BATCH_SIZE))
    logger.info("epsilon: {}".format(epsilon))
    logger.info("delta: {}".format(delta))
    logger.info("device: {}".format(device_name))

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        logger.info("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    else:
        logger.info("CUDA_VISIBLE_DEVICES not defined.")
    logger.info("CPUs: {}".format(os.sched_getaffinity(0)))
    logger.info("GPUs: {}".format(torch.cuda.device_count()))
    logger.info("Hostname: {}".format(socket.gethostname()))

    root_path = '{}'.format(expname)
    if not os.path.exists(root_path):
        os.mkdir(root_path)
    save_path = root_path

    print('root_path:', root_path)
    print('save_path:', save_path)

    lr = 0.1
    num_dummy = 100
    Iteration = 901
    num_exp = 1

    device = torch.device(device_name)

    trainDatasets, testDataset, fulltrainDataset = getSTDataset(numberClients)
    print(trainDatasets)
    trainLoaders = [DataLoader(dataset, BATCH_SIZE, True) for dataset in trainDatasets]
    testLoader = DataLoader(testDataset, BATCH_SIZE, False)
    fulltrainLoader = DataLoader(fulltrainDataset, BATCH_SIZE, False)
    mean_expression, mean_expression_tumor, mean_expression_normal, median_expression = compute_mean_expression()
    loader=testLoader

    # Find number of required outputs
    if args.task == "tumor":
        outputs = 2
    elif args.task == "gene":
        outputs = fulltrainDataset[0][2].shape[0]
    elif args.task == "geneb":
        outputs = 2 * fulltrainDataset[0][2].shape[0]
    elif args.task == "count":
        outputs = 1

    ''' train DLG and iDLG '''

    iii = 0
    for idx_net in range(num_exp):
        for method in ['iDLG']:
            with tqdm(total=len(loader)) as tq:

                # for each idx, we initial the model
                net = LeNet()

                print('running %d|%d experiment' % (idx_net, num_exp))
                net = net.to(device)

                for X, y, gene, c, ind, pat, s, pix, f in tqdm(loader):
                    print('iDLG on idx: {}'.format(iii))

                    gt_data = X.to(device)
                    gene = gene.to(device)

                    print(gt_data.shape,gene.shape)

                    criterion = nn.CrossEntropyLoss().to(device)

                    # compute original gradient
                    pred = net(gt_data)
                    y = torch.sum((pred - gene) ** 2) / outputs

                    dy_dx = torch.autograd.grad(y, net.parameters())
                    original_dy_dx = list((_.detach().clone() for _ in dy_dx))

                    if mode=='SGD':
                        pass
                    elif mode=='DP':
                        aggregated = []
                        for i in range(len(original_dy_dx)):
                            aggregated.append(dp(original_dy_dx[i], device=device, epsilon=epsilon, delta=delta))
                        original_dy_dx=aggregated

                    # generate dummy data and label
                    dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)

                    optimizer = torch.optim.Adam([dummy_data, ], lr=lr)
                    # predict the ground-truth label

                    history = []
                    history_iters = []
                    losses = []
                    mses = []
                    train_iters = []

                    print('lr =', lr)
                    for iters in range(Iteration):
                        print('idx: {}, {}/{}'.format(iii,iters,Iteration))
                        def closure():
                            optimizer.zero_grad()
                            pred = net(dummy_data)
                            dummy_loss = torch.sum((pred - gene) ** 2) / outputs
                            dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)

                            grad_diff = 0
                            for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                                grad_diff += ((gx - gy) ** 2).sum()
                            grad_diff.backward()
                            return grad_diff

                        optimizer.step(closure)
                        current_loss = closure().item()
                        train_iters.append(iters)
                        losses.append(current_loss)
                        mses.append(torch.mean((dummy_data - gt_data) ** 2).item())

                        current_time = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
                        print(current_time, iters, 'loss = %.8f, mse = %.8f' % (current_loss, mses[-1]))

                        if iters % int(100) == 0 and iters>=100:
                            history.append(np.transpose(dummy_data.detach().cpu().numpy().squeeze(),(1,2,0)))
                            history_iters.append(iters)

                            plt.figure(figsize=(12, 8),dpi=300)
                            plt.subplot(3, 10, 1)
                            plt.imshow(np.transpose(gt_data.detach().cpu().numpy().squeeze(),(1,2,0)))
                            plt.title('original')
                            plt.axis('off')
                            for j in range(min(len(history), 29)):
                                plt.subplot(3, 10, j + 2)
                                plt.imshow(history[j])
                                plt.title('iter=%d' % (history_iters[j]))
                                plt.axis('off')
                            if method == 'iDLG':
                                plt.savefig('%s/iDLG_on_%s_%05d.png' % (save_path, iii, iters))
                                plt.close()

                            if current_loss < 0.0000001:  # converge
                                break
                    iii+=1
                    if iii>num_dummy:
                        break
