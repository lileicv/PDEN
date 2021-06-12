
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import os
import numpy as np
import click
import pandas as pd

from network import mnist_net
import data_loader
from main_base import evaluate
from utils import log

@click.command()
@click.option('--gpu', type=str, default='0', help='选择GPU编号')
@click.option('--modelpath', type=str, default='saved/best.pkl')
@click.option('--svpath', type=str, default=None, help='保存日志的路径')
@click.option('--channels', type=int, default=3)
def main(gpu, modelpath, svpath, channels):
    evaluate_digit(gpu, modelpath, svpath, channels)
    
def evaluate_digit(gpu, modelpath, svpath, channels=3):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    # 加载模型
    if channels == 3:
        cls_net = mnist_net.ConvNet().cuda()
    elif channels == 1:
        cls_net = mnist_net.ConvNet(imdim=channels).cuda()
    saved_weight = torch.load(modelpath)
    cls_net.load_state_dict(saved_weight['cls_net'])
    #cls_net.eval()

    # 测试
    str2fun = { 
        'mnist': data_loader.load_mnist,
        'mnist_m': data_loader.load_mnist_m,
        'usps': data_loader.load_usps,
        'svhn': data_loader.load_svhn,
        'syndigit': data_loader.load_syndigit,
        }   
    columns = ['mnist', 'mnist_m', 'usps', 'svhn', 'syndigit']
    rst = []
    for data in columns:
        teset = str2fun[data]('test', channels=channels)
        teloader = DataLoader(teset, batch_size=128, num_workers=8)
        teacc = evaluate(cls_net, teloader)
        rst.append(teacc)
    
    df = pd.DataFrame([rst], columns=columns)
    print(df)
    if svpath is not None:
        df.to_csv(svpath)

if __name__=='__main__':
    main()

