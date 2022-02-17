import datetime
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from easydl import *
from PIL import Image
import logging
import argparse

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

def Process(im_flatten):
    maxValue = torch.max(im_flatten)
    minValue = torch.min(im_flatten)
    im_flatten = im_flatten-minValue
    im_flatten = im_flatten/(maxValue-minValue)
    return im_flatten


def Attack(mynet, target_label,device):
    target_label=torch.tensor(target_label)
    aim_flatten = torch.zeros(1, 20531).to(device)
    v = torch.zeros(1, 20531).to(device)
    aim_flatten.requires_grad = True
    costn_1 = 10
    b = 0
    g = 0
    out = mynet.forward(aim_flatten.detach())
    after_softmax = F.softmax(out, dim=-1)
    predict = torch.argmax(after_softmax)
    #print(predict)
    for i in range(alpha):
        out = mynet.forward(aim_flatten)
        if aim_flatten.grad is not None:
            aim_flatten.grad.zero_()
        out = out.reshape(1, classes)
        target_class = torch.tensor([target_label]).to(device)
        cost = nn.CrossEntropyLoss()(out, target_class)
        cost.backward()
        aim_grad = aim_flatten.grad
        # see https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD
        aim_flatten = aim_flatten-learning_rate*(momentum*v+aim_grad)
        aim_flatten = Process(aim_flatten)
        aim_flatten = torch.clamp(aim_flatten.detach(), 0, 1)
        aim_flatten.requires_grad = True
        #print(i,cost)
        logger.info('{}/{}: {}'.format(i,alpha,cost.detach().cpu().numpy()))
        if cost >= costn_1:
            b = b+1
            if b > beta:
                break
        else:
            b = 0
        costn_1 = cost
        if cost < gama:
            break
    out = mynet.forward(aim_flatten.detach())
    after_softmax = F.softmax(out, dim=-1)
    predict = torch.argmax(after_softmax)
    print(predict,cost)
    oim_flatten = aim_flatten.detach().cpu().numpy()
    with open(f'{log_dir}/{target_label}.npy', 'wb') as f:
        np.save(f, oim_flatten)

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

cancerTypes = ['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'COADREAD', 'DLBC', 'ESCA', 'GBM', 'GBMLGG', 'HNSC',
                   'KICH', 'KIPAN', 'KIRC', 'KIRP', 'LAML', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'MESO', 'OV', 'PAAD', 'PCPG',
                   'PRAD', 'READ', 'SARC', 'SKCM', 'STAD', 'STES', 'TGCT', 'THCA', 'THYM', 'UCEC', 'UCS', 'UVM']

parser = argparse.ArgumentParser(description='dfsa')
parser.add_argument('--model', default='SGD_data0_rep0_modelbest.tar')
parser.add_argument('--lr',type=float,default=0.1)
args = parser.parse_args()
model_id=args.model
model_id=model_id.split('/')[1]

gpus = 0
data_workers = 0
# batch_size = 64
classes = len(cancerTypes)
root_dir = "MIA_CC/"
#model_id='SGD_data0_rep0_modelbest.tar'
model_weight = "model/{}".format(model_id)

alpha = 10000
beta = 100
gama = 0.001
learning_rate = args.lr
momentum = 0.9

cudnn.benchmark = True
cudnn.deterministic = True
seed = 9970
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device='cuda:0'

log_dir = f'{root_dir}/{model_id}'
logger = log_creater(output_dir=log_dir, expname=model_id)

logger.info("model: {}".format(model_id))

net = Model(input_size=20531,output_size=len(cancerTypes))
mynet = nn.DataParallel(net, output_device=device).train(False)
mynet=mynet.to(device)
assert os.path.exists(model_weight)
mynet.load_state_dict(torch.load(open(model_weight, 'rb')))
for i in range(classes):
    logger.info(f'---class{i}---')
    Attack(mynet=mynet, target_label=i,device=device)
