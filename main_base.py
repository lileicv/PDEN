
'''
训练 base 模型
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, RandomSampler
from torchvision import models
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import os
import click
import time
import numpy as np

from network import mnist_net
import data_loader

HOME = os.environ['HOME']

@click.command()
@click.option('--gpu', type=str, default='0', help='选择gpu')
@click.option('--data', type=str, default='mnist', help='数据集名称')
@click.option('--ntr', type=int, default=None, help='选择训练集前ntr个样本')
@click.option('--translate', type=float, default=None, help='随机平移数据增强')
@click.option('--autoaug', type=str, default=None, help='AA FastAA RA')
@click.option('--epochs', type=int, default=100)
@click.option('--nbatch', type=int, default=None, help='每个epoch中batch的数量')
@click.option('--batchsize', type=int, default=256, help='每个batch中样本的数量')
@click.option('--lr', type=float, default=1e-3)
@click.option('--lr_scheduler', type=str, default='none', help='是否选择学习率衰减策略')
@click.option('--svroot', type=str, default='./saved', help='项目文件保存路径')
def experiment(gpu, data, ntr, translate, autoaug, epochs, nbatch, batchsize, lr, lr_scheduler, svroot):
    settings = locals().copy()
    print(settings)

    # 全局设置
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    if not os.path.exists(svroot):
        os.makedirs(svroot)
    writer = SummaryWriter(svroot)
    
    # 加载数据集和模型
    if data in ['mnist', 'mnist_t']:
        # 加载数据集
        if data == 'mnist':
            trset = data_loader.load_mnist('train', translate=translate, ntr=ntr, autoaug=autoaug)
        elif data == 'mnist_t':
            trset = data_loader.load_mnist_t('train', translate=translate, ntr=ntr)
        teset = data_loader.load_mnist('test')
        trloader = DataLoader(trset, batch_size=batchsize, num_workers=8, \
                sampler=RandomSampler(trset, True, nbatch*batchsize))
        teloader = DataLoader(teset, batch_size=batchsize, num_workers=8, shuffle=False)
        cls_net = mnist_net.ConvNet().cuda()
        cls_opt = optim.Adam(cls_net.parameters(), lr=lr)
    
    elif data == 'mnistvis':
        trset = data_loader.load_mnist('train')
        teset = data_loader.load_mnist('test')
        trloader = DataLoader(trset, batch_size=batchsize, num_workers=8, \
                sampler=RandomSampler(trset, True, nbatch*batchsize))
        teloader = DataLoader(teset, batch_size=batchsize, num_workers=8, shuffle=False)
        cls_net= mnist_net.ConvNetVis().cuda()
        cls_opt = optim.Adam(cls_net.parameters(), lr=lr)
    
    elif data == 'cifar10':
        # 加载数据集
        trset = data_loader.load_cifar10(split='train', autoaug=autoaug)
        teset = data_loader.load_cifar10(split='test')
        trloader = DataLoader(trset, batch_size=batchsize, num_workers=8, shuffle=True, drop_last=True)
        teloader = DataLoader(teset, batch_size=batchsize, num_workers=8, shuffle=False)
        cls_net = wideresnet.WideResNet(16, 10, 4).cuda()
        cls_opt = optim.SGD(cls_net.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
        if lr_scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(cls_opt, epochs)
    elif 'synthia' in data:
        # 加载数据集
        branch = data.split('_')[1]
        trset = data_loader.load_synthia(branch)
        trloader = DataLoader(trset, batch_size=batchsize, num_workers=8, shuffle=True)
        teloader = DataLoader(trset, batch_size=batchsize, num_workers=8, shuffle=True)
        imsize = [192, 320]
        nclass = 14
        # 加载模型
        cls_net = fcn.FCN_resnet50(nclass=nclass).cuda()
        cls_opt = optim.Adam(cls_net.parameters(), lr=lr)#, weight_decay=1e-4) # 对于synthia 加上weigh_decay会掉1-2个点
        if lr_scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(cls_opt, epochs*len(trloader))
    
    cls_criterion = nn.CrossEntropyLoss()

    # 开始训练
    best_acc = 0
    for epoch in range(epochs):
        t1 = time.time()
        
        loss_list = []
        cls_net.train()
        for i, (x, y) in enumerate(trloader):
            x, y = x.cuda(), y.cuda()
            
            # 训练
            p = cls_net(x)
            cls_loss = cls_criterion(p, y)
            #torch.cuda.synchronize()
            cls_opt.zero_grad()
            cls_loss.backward()
            cls_opt.step()
            
            loss_list.append([cls_loss.item()])
            
            # 调整学习率
            if lr_scheduler in ['cosine']:
                scheduler.step()

        cls_loss, = np.mean(loss_list, 0)
        

        # 测试，并保存最优模型
        cls_net.eval()
        if data in ['mnist', 'mnist_t', 'cifar10', 'mnistvis']:
            teacc = evaluate(cls_net, teloader)
        elif 'synthia' in data:
            teacc = evaluate_seg(cls_net, teloader, nclass) # 这里算的其实是 miou

        if best_acc < teacc:
            best_acc = teacc
            torch.save({'cls_net':cls_net.state_dict()}, os.path.join(svroot, 'best.pkl'))

        # 保存日志
        t2 = time.time()
        print(f'epoch {epoch}, time {t2-t1:.2f}, cls_loss {cls_loss:.4f} teacc {teacc:2.2f}')
        writer.add_scalar('scalar/cls_loss', cls_loss, epoch)
        writer.add_scalar('scalar/teacc', teacc, epoch)

    writer.close()

def evaluate(net, teloader):
    correct, count = 0, 0
    ps = []
    ys = []
    for i,(x1, y1) in enumerate(teloader):
        with torch.no_grad():
            x1 = x1.cuda()
            p1 = net(x1)
            p1 = p1.argmax(dim=1)
            ps.append(p1.detach().cpu().numpy())
            ys.append(y1.numpy())
    # 计算评价指标
    ps = np.concatenate(ps)
    ys = np.concatenate(ys)
    acc = np.mean(ys==ps)*100
    return acc

if __name__=='__main__':
    experiment()

