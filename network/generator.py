
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AdaIN2d(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)
    def forward(self, x, s): 
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta
        #return (1+gamma)*(x)+beta

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
    def forward(self, x):
        return x.view((x.size(0),)+self.shape)

class cnnGenerator(nn.Module):
    def __init__(self, n=16, kernelsize=3, imdim=3, imsize=[192, 320]):
        ''' w_ln 局部噪声权重
        '''
        super().__init__()
        stride = (kernelsize-1)//2
        self.zdim = zdim = 10
        self.imdim = imdim
        self.imsize = imsize

        self.conv1 = nn.Conv2d(imdim, n, kernelsize, 1, stride)
        self.conv2 = nn.Conv2d(n, 2*n, kernelsize, 1, stride)
        self.adain2 = AdaIN2d(zdim, 2*n)
        self.conv3 = nn.Conv2d(2*n, 4*n, kernelsize, 1, stride)
        self.conv4 = nn.Conv2d(4*n, imdim, kernelsize, 1, stride)

    def forward(self, x, rand=False): 
        ''' x '''
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        if rand:
            z = torch.randn(len(x), self.zdim).cuda()
            x = self.adain2(x, z)
        x = F.relu(self.conv3(x))
        x = torch.sigmoid(self.conv4(x))
        return x

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
    def forward(self, x): 
        return x.view((x.size(0),)+self.shape)

class stnGenerator(nn.Module):
    ''' 仿射变换 '''
    def __init__(self, zdim=10, imsize=[32,32], mode=None):
        super().__init__()
        self.mode = mode
        self.zdim = zdim
        
        self.mapz = nn.Linear(zdim, imsize[0]*imsize[1])
        if imsize == [32,32]:
            self.loc = nn.Sequential(
                    nn.Conv2d( 4,  16, 5), nn.MaxPool2d(2), nn.ReLU(),
                    nn.Conv2d( 16, 32, 5), nn.MaxPool2d(2), nn.ReLU(),)
            self.fc_loc = nn.Sequential(
                    nn.Linear(32*5*5, 32), nn.ReLU(),
                    nn.Linear(32, 6))
        # init the weight
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1,0,0,0,1,0]))
    def forward(self, x, rand, return_H=False):
        if rand:
            z = torch.randn(len(x), self.zdim).cuda()
        z = self.mapz(z).view(len(x), 1, x.size(2), x.size(3))
        loc = self.loc(torch.cat([x, z], dim=1)) # [N, -1]
        loc = loc.view(len(loc), -1)
        H = self.fc_loc(loc)
        H = H.view(len(H), 2, 3)
        if self.mode == 'translate':
            H[:,0,0] = 1 
            H[:,0,1] = 0 
            H[:,1,0] = 0 
            H[:,1,1] = 1 
        grid = F.affine_grid(H, x.size())
        x = F.grid_sample(x, grid)
        if return_H:
            return x, H
        else:
            return x

if __name__=='__main__':
    x = torch.ones(4, 3, 32, 32)
    z = torch.ones(4, 10)
    
    g = stnGenerator(10, [32, 32])
    y = g(x, z)


