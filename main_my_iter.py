
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

from con_losses import SupConLoss
from network import mnist_net, generator
import data_loader
from main_base import evaluate

HOME = os.environ['HOME']

@click.command()
@click.option('--gpu', type=str, default='0', help='选择gpu')
@click.option('--data', type=str, default='mnist', help='数据集名称')
@click.option('--ntr', type=int, default=None, help='选择训练集前ntr个样本')
@click.option('--gen', type=str, default='cnn', help='cnn/hr')
@click.option('--gen_mode', type=str, default=None, help='生成器模式')
@click.option('--n_tgt', type=int, default=10, help='学习多少了tgt模型')
@click.option('--tgt_epochs', type=int, default=10, help='每个目标域训练多少了epochs')
@click.option('--tgt_epochs_fixg', type=int, default=None, help='当epoch大于该值，将G fix掉')
@click.option('--nbatch', type=int, default=None, help='每个epoch中包含多少了batch')
@click.option('--batchsize', type=int, default=256)
@click.option('--lr', type=float, default=1e-3)
@click.option('--lr_scheduler', type=str, default='none', help='是否选择学习率衰减策略')
@click.option('--svroot', type=str, default='./saved')
@click.option('--ckpt', type=str, default='./saved/best.pkl')
@click.option('--w_cls', type=float, default=1.0, help='cls项权重')
@click.option('--w_info', type=float, default=1.0, help='infomin项权重')
@click.option('--w_cyc', type=float, default=10.0, help='cycleloss项权重')
@click.option('--w_div', type=float, default=1.0, help='多形性loss权重')
@click.option('--div_thresh', type=float, default=0.1, help='div_loss 阈值')
@click.option('--w_tgt', type=float, default=1.0, help='target domain样本更新 tasknet 的强度控制')
@click.option('--interpolation', type=str, default='pixel', help='在源域和生成域之间插值得到新的域，两种方式：img/pixel')
def experiment(gpu, data, ntr, gen, gen_mode, \
        n_tgt, tgt_epochs, tgt_epochs_fixg, nbatch, batchsize, lr, lr_scheduler, svroot, ckpt, \
        w_cls, w_info, w_cyc, w_div, div_thresh, w_tgt, interpolation):
    settings = locals().copy()
    print(settings)

    # 全局设置
    zdim = 10
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    g1root = os.path.join(svroot, 'g1')
    if not os.path.exists(g1root):
        os.makedirs(g1root)
    writer = SummaryWriter(svroot)
    
    # 加载数据集
    imdim = 3 # 默认3通道
    if data in ['mnist', 'mnist_t', 'mnistvis']:
        if data in [ 'mnist', 'mnistvis']:
            trset = data_loader.load_mnist('train', ntr=ntr)
            teset = data_loader.load_mnist('test')
        elif data == 'mnist_t':
            trset = data_loader.load_mnist_t('train', ntr=ntr)
            teset = data_loader.load_mnist('test')
        imsize = [32, 32]
    trloader = DataLoader(trset, batch_size=batchsize, num_workers=8, \
                sampler=RandomSampler(trset, True, nbatch*batchsize))
    teloader = DataLoader(teset, batch_size=batchsize, num_workers=8, shuffle=False)
    
    # 加载模型
    def get_generator(name):
        if name=='cnn':
            g1_net = generator.cnnGenerator(imdim=imdim, imsize=imsize).cuda()
            g2_net = generator.cnnGenerator(imdim=imdim, imsize=imsize).cuda()
            g1_opt = optim.Adam(g1_net.parameters(), lr=lr)
            g2_opt = optim.Adam(g2_net.parameters(), lr=lr)
        elif gen=='hr':
            1/0
            g1_net = hrnet.HRGenerator(zdim=zdim).cuda()
            g2_net = hrnet.HRGenerator(zdim=zdim).cuda()
            g1_opt = optim.Adam(g1_net.parameters(), lr=lr)
            g2_opt = optim.Adam(g2_net.parameters(), lr=lr)
        elif gen=='stn':
            g1_net = generator.stnGenerator(zdim=zdim, mode=gen_mode).cuda()
            g2_net = None
            g1_opt = optim.Adam(g1_net.parameters(), lr=lr/2)
            g2_opt = None
        return g1_net, g2_net, g1_opt, g2_opt

    g1_list = []
    if data in ['mnist', 'mnist_t']:
        src_net = mnist_net.ConvNet().cuda()
        saved_weight = torch.load(ckpt)
        src_net.load_state_dict(saved_weight['cls_net'])
        src_opt = optim.Adam(src_net.parameters(), lr=lr)

    elif data == 'mnistvis':
        src_net = mnist_net.ConvNetVis().cuda()
        saved_weight = torch.load(ckpt)
        src_net.load_state_dict(saved_weight['cls_net'])
        src_opt = optim.Adam(src_net.parameters(), lr=lr)

    cls_criterion = nn.CrossEntropyLoss()
    con_criterion = SupConLoss()

    # 开始训练
    global_best_acc = 0
    for i_tgt in range(n_tgt):
        print(f'target domain {i_tgt}')

        ####################### 学习第i个tgt generator
        if lr_scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(src_opt, tgt_epochs*len(trloader))
        g1_net, g2_net, g1_opt, g2_opt = get_generator(gen)
        best_acc = 0
        for epoch in range(tgt_epochs):
            t1 = time.time()
            
            # 如果 flag_fixG = False, 锁定 G
            #      flag_fixG = True, 更新 G
            flag_fixG = False
            if (tgt_epochs_fixg is not None) and (epoch >= tgt_epochs_fixg):
                flag_fixG = True
            loss_list = []
            time_list = []
            #src_net.train()
            src_net.eval()
            for i, (x, y) in enumerate(trloader):
                x, y = x.cuda(), y.cuda()

                # 增强新数据
                if len(g1_list)>0: # 如果生成器
                    idx = np.random.randint(0, len(g1_list))
                    #rand = torch.randn(len(x), zdim).cuda()
                    if gen in ['hr', 'cnn']:
                        with torch.no_grad():
                            x2_src = g1_list[idx](x, rand=True)
                        # domain 插值
                        if interpolation == 'img':
                            rand = torch.rand(len(x), 1, 1, 1).cuda()
                            x3_mix = rand*x + (1-rand)*x2_src
                    elif gen == 'stn':
                        with torch.no_grad():
                            x2_src, H = g1_list[idx](x, rand=True, return_H=True)
                        # domain 插值
                        if interpolation == 'H':
                            rand = torch.rand(len(x), 1, 1).cuda()
                            std_H = torch.tensor([[1, 0, 0], [0, 1, 0]]).float().cuda()
                            H = rand*std_H + (1-rand)*H
                            grid = F.affine_grid(H, x.size())
                            x3_mix = F.grid_sample(x, grid)

                # 合成新数据
                #rand = torch.randn(len(x), zdim).cuda()
                #rand2 = torch.randn(len(x), zdim).cuda()
                if gen in ['cnn', 'hr']:
                    x_tgt = g1_net(x, rand=True)
                    x2_tgt = g1_net(x, rand=True)
                elif gen == 'stn':
                    x_tgt, H_tgt = g1_net(x, rand=True, return_H=True)
                    x2_tgt, H2_tgt = g1_net(x, rand=True, return_H=True)

                # 前向传播
                p1_src, z1_src = src_net(x, mode='train')
                if len(g1_list)>0: # 如果生成器
                    p2_src, z2_src = src_net(x2_src, mode='train')
                    p3_mix, z3_mix = src_net(x3_mix, mode='train')
                    zsrc = torch.cat([z1_src.unsqueeze(1), z2_src.unsqueeze(1), z3_mix.unsqueeze(1)], dim=1)
                    src_cls_loss = cls_criterion(p1_src, y) + cls_criterion(p2_src, y) + cls_criterion(p3_mix, y)
                else:
                    zsrc = z1_src.unsqueeze(1)
                    src_cls_loss = cls_criterion(p1_src, y)
                p_tgt, z_tgt = src_net(x_tgt, mode='train')
                tgt_cls_loss = cls_criterion(p_tgt, y)

                # 更新 src_net
                zall = torch.cat([z_tgt.unsqueeze(1), zsrc], dim=1)
                con_loss = con_criterion(zall, adv=False)
                loss = src_cls_loss + w_tgt*con_loss + w_tgt*tgt_cls_loss # w_tgt 默认 1.0
                src_opt.zero_grad()
                if flag_fixG:
                    loss.backward()
                else:
                    loss.backward(retain_graph=True)
                src_opt.step()

                # 更新 g1_net
                if flag_fixG:
                    # fix G，只训练 tasknet
                    con_loss_adv = torch.tensor(0)
                    div_loss = torch.tensor(0)
                    cyc_loss = torch.tensor(0)
                else:
                    idx = np.random.randint(0, zsrc.size(1))
                    zall = torch.cat([z_tgt.unsqueeze(1), zsrc[:,idx:idx+1].detach()], dim=1)
                    con_loss_adv = con_criterion(zall, adv=True)
                    if gen in ['cnn', 'hr']:
                        div_loss = (x_tgt-x2_tgt).abs().mean([1,2,3]).clamp(max=div_thresh).mean() # 约束生成器散度
                        x_tgt_rec = g2_net(x_tgt)
                        cyc_loss = F.mse_loss(x_tgt_rec, x)
                    elif gen == 'stn':
                        div_loss = (H_tgt-H2_tgt).abs().mean([1,2]).clamp(max=div_thresh).mean()
                        cyc_loss = torch.tensor(0).cuda()
                    loss = w_cls*tgt_cls_loss - w_div*div_loss + w_cyc*cyc_loss + w_info*con_loss_adv
                    g1_opt.zero_grad()
                    if g2_opt is not None:
                        g2_opt.zero_grad()
                    loss.backward()
                    g1_opt.step()
                    if g2_opt is not None:
                        g2_opt.step()
                # 更新学习率
                if lr_scheduler in ['cosine']:
                    scheduler.step()
               
                loss_list.append([src_cls_loss.item(), tgt_cls_loss.item(), con_loss.item(), con_loss_adv.item(), div_loss.item(), cyc_loss.item()])
            src_cls_loss, tgt_cls_loss, con_loss, con_loss_adv, div_loss, cyc_loss = np.mean(loss_list, 0)
            
            # 测试
            src_net.eval()
            # mnist、cifar的测试过程和 synthia不一样
            if data in ['mnist', 'mnist_t', 'mnistvis']:
                teacc = evaluate(src_net, teloader)
            if best_acc < teacc:
                best_acc = teacc
                torch.save({'cls_net':src_net.state_dict()}, os.path.join(svroot, f'{i_tgt}-best.pkl'))
            #if global_best_acc < teacc:
            #    global_best_acc = teacc
            #    torch.save({'cls_net':src_net.state_dict()}, os.path.join(svroot, f'best.pkl'))

            t2 = time.time()

            # 保存日志
            print(f'epoch {epoch}, time {t2-t1:.2f}, src_cls {src_cls_loss:.4f} tgt_cls {tgt_cls_loss:.4f} con {con_loss:.4f} con_adv {con_loss_adv:.4f} div {div_loss:.4f} cyc {cyc_loss:.4f} /// teacc {teacc:2.2f}')
            writer.add_scalar('scalar/src_cls_loss', src_cls_loss, i_tgt*tgt_epochs+epoch)
            writer.add_scalar('scalar/tgt_cls_loss', tgt_cls_loss, i_tgt*tgt_epochs+epoch)
            writer.add_scalar('scalar/con_loss', con_loss, i_tgt*tgt_epochs+epoch)
            writer.add_scalar('scalar/con_loss_adv', con_loss_adv, i_tgt*tgt_epochs+epoch)
            writer.add_scalar('scalar/div_loss', div_loss, i_tgt*tgt_epochs+epoch)
            writer.add_scalar('scalar/cyc_loss', cyc_loss, i_tgt*tgt_epochs+epoch)
            writer.add_scalar('scalar/teacc', teacc, i_tgt*tgt_epochs+epoch)

            g1_all = g1_list + [g1_net]
            x = x[0:10]
            l1 = make_grid(x, 1, 2, pad_value=128)
            l_list = [l1]
            with torch.no_grad():
                for i in range(len(g1_all)):
                    #rand = torch.randn(len(x), zdim).cuda()
                    x_ = g1_all[i](x, rand=True)
                    l_list.append(make_grid(x_, 1, 2, pad_value=128))
                if g2_net is not None:
                    x_x = g2_net(x_)
                    l_list.append(make_grid(x_x, 1, 2, pad_value=128))
                rst = make_grid(torch.stack(l_list), len(l_list), pad_value=128)
                writer.add_image('im-gen', rst, i_tgt*tgt_epochs+epoch)

                x_copy = x[0:1].repeat(16, 1, 1, 1)
                #rand = torch.randn(16, zdim).cuda()
                x_copy_ = g1_net(x_copy, rand=True)
                rst = make_grid(x_copy_, 4, 2, pad_value=128)
                writer.add_image('im-div', rst, i_tgt*tgt_epochs+epoch)
            if len(g1_list)>0:
                l1 = make_grid(x[0:6], 6, 2, pad_value=128)
                l2 = make_grid(x2_src[0:6], 6, 2, pad_value=128)
                l3 = make_grid(x3_mix[0:6], 6, 2, pad_value=128)
                rst = make_grid(torch.stack([l1, l3, l2]), 1, pad_value=128)
                writer.add_image('im-mix', rst, i_tgt*tgt_epochs+epoch)

        # 保存训练好的G1
        torch.save({'g1':g1_net.state_dict()}, os.path.join(g1root, f'{i_tgt}.pkl'))
        g1_list.append(g1_net)

        # 测试 i_tgt 模型的泛化效果
        from main_test_digit import evaluate_digit
        if data == 'mnist':
            pklpath = f'{svroot}/{i_tgt}-best.pkl'
            evaluate_digit(gpu, pklpath, pklpath+'.test')

    writer.close()

if __name__=='__main__':
    experiment()

