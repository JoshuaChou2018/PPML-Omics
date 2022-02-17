#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021-12-12 20:06
# @Author  : Juexiao Zhou & Siyuan Chen
# @Site    : 
# @File    : Test.py
# @Software: PyCharm
import os
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import copy
import logging
from scipy import stats
from tqdm import tqdm
from sklearn.cluster import KMeans
import metrics as M
from collections import OrderedDict
import argparse
import torch
import torch.nn as nn
root = os.path.dirname(__file__)


os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def log_creater(output_dir, expname):
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


class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model,self).__init__()

        self.hidden1 = nn.Linear(input_size,1024)
        self.hidden2 = nn.Linear(1024, 16)
        self.hidden3 = nn.Linear(16, 1024)
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
        self.label_dic =  dict(sorted(self.label_dic.items()))

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
        #label = data_dic['cancertype']
        label=  data_dic['celltype']

        # Genes: 20531 len Tensor
        # lable: float value
        # info:  dic info
        return torch.Tensor(genes), label

    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small batch size of sets.
        return len(self.samples)
if __name__ == '__main__':
    if __name__ == '__main__':
        parser = argparse.ArgumentParser(description='dfsa')
        parser.add_argument('--dataset', default="yan", help='dataset name')
        parser.add_argument('--model', default="", help='model path')
        parser.add_argument('--expname', default="", help='expname')

        args = parser.parse_args()
        dataset_name = args.dataset
        model_path = args.model
        expname = args.expname
        device_name = 'cuda:0'

        BATCH_SIZE = 1

        device = torch.device(device_name)

        dataset = Single_Cell_Data(dataset_name=dataset_name)
        k = len(dataset.label_dic.keys())
        input_size = dataset.feature_size

        print("k = {}".format(k))
        print("input_size = {}".format(input_size))

        data_loader = DataLoader(dataset, BATCH_SIZE, True)



        model = Model(input_size=input_size,
                      output_size=input_size)

        print("=> Loading {}".format(model_path))
        model = model.to(device)
        model = torch.nn.DataParallel(model).cuda()

        print("=> Loading module from {}".format(model_path))
        model.load_state_dict(torch.load(model_path))

        logger = log_creater(output_dir=os.path.join(root,
                                                     "result",
                                                     "SC",
                                                     expname),
                             expname="{}_{}".format(expname,dataset_name))

        print("=> Aggregate Features...")

        model.eval()


        final_metric = {
            "ARI":[],
            "NMI": [],
            "CA": [],
            "JI": [],
        }
        final_output = None
        best_ARI = 0
        best_output = None

        for i in range(5):
            labels = []
            features = []
            for genes, label in tqdm(data_loader):
                genes, label = genes.to(device), label.to(device)
                feat, out = model(genes)
                labels += list(label.detach().cpu().numpy())
                features += list(feat.detach().cpu().numpy())
            print("feature shape: {}".format(np.array(features).shape))

            kmeans = KMeans(n_clusters=k, random_state=0).fit(np.array(features))

            pred = kmeans.labels_

            metrics = M.compute_metrics(labels, pred)

            output = {
                "feature": np.array(features),
                "label": labels,
                "pred": pred
            }


            logger.info('Test iter {} ARI:{} \t NMI:{} \t CA:{} \t JI:{}'.format(i,
                                                                 metrics['ARI'],
                                                                 metrics['NMI'],
                                                                 metrics['CA'],
                                                                 metrics['JI'], ))

            if metrics["ARI"] > best_ARI:
                best_ARI = metrics["ARI"]
                final_output = output.copy()

            final_metric["ARI"].append(metrics["ARI"])
            final_metric["NMI"].append(metrics["NMI"])
            final_metric["CA"].append(metrics["CA"])
            final_metric["JI"].append(metrics["JI"])
        logger.info("************************************************************")
        logger.info('Final Best: ARI:{} \t NMI:{} \t CA:{} \t JI:{}'.format(
                                         np.max(final_metric['ARI']),
                                         np.max(final_metric['NMI']),
                                         np.max(final_metric['CA']),
                                         np.max(final_metric['JI']) ))



        output_file = os.path.join(root,
                                   "result",
                                   "SC",
                                   expname,
                                   "{}_{}.npy".format(expname, dataset_name))
        final_output["label_dic"] = dataset.label_dic

        np.save(output_file, final_output)
