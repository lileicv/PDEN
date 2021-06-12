''' Digit 实验
'''
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset
from torchvision import transforms
from torchvision.datasets import MNIST, USPS, SVHN, CIFAR10, STL10

import os
import pickle
import numpy as np
from scipy.io import loadmat
from PIL import Image

from tools.autoaugment import SVHNPolicy, CIFAR10Policy
from tools.randaugment import RandAugment

class myTensorDataset(Dataset):
    def __init__(self, x, y, transform=None, twox=False):
        self.x = x
        self.y = y
        self.transform = transform
        self.twox = twox
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        if self.transform is not None:
            x = self.transform(x)
            if self.twox:
                x2 = self.transform(x)
                return (x, x2), y
        return x, y

HOME = os.environ['HOME']

def resize_imgs(x, size):
    ''' 目前只能处理单通道 
        x [n, 28, 28]
        size int
    '''
    resize_x = np.zeros([x.shape[0], size, size])
    for i, im in enumerate(x):
        im = Image.fromarray(im)
        im = im.resize([size, size], Image.ANTIALIAS)
        resize_x[i] = np.asarray(im)
    return resize_x

def load_mnist(split='train', translate=None, twox=False, ntr=None, autoaug=None, channels=3):
    '''
        autoaug == 'AA', AutoAugment
                   'FastAA', Fast AutoAugment
                   'RA', RandAugment
        channels == 3 默认返回 rgb 3通道图像
                    1 返回单通道图像
    '''
    path = f'data/mnist-{split}.pkl'
    if not os.path.exists(path):
        dataset = MNIST(f'{HOME}/.pytorch/MNIST', train=(split=='train'), download=True)
        x, y = dataset.data, dataset.targets
        if split=='train':
            x, y = x[0:10000], y[0:10000]
        x = torch.tensor(resize_imgs(x.numpy(), 32))
        x = (x.float()/255.).unsqueeze(1).repeat(1,3,1,1)
        with open(path, 'wb') as f:
            pickle.dump([x, y], f)
    with open(path, 'rb') as f:
        x, y = pickle.load(f)
        if channels == 1:
            x = x[:,0:1,:,:]
    
    if ntr is not None:
        x, y = x[0:ntr], y[0:ntr]
    
    # 如果没有数据增强
    if (translate is None) and (autoaug is None):
        dataset = TensorDataset(x, y)
        return dataset
    
    # 数据增强管道
    transform = [transforms.ToPILImage()]
    if translate is not None:
        transform.append(transforms.RandomAffine(0, [translate, translate]))
    if autoaug is not None:
        if autoaug == 'AA':
            transform.append(SVHNPolicy())
        elif autoaug == 'RA':
            transform.append(RandAugment(3,4))
    transform.append(transforms.ToTensor())
    transform = transforms.Compose(transform)
    dataset = myTensorDataset(x, y, transform=transform, twox=twox)
    return dataset

def load_usps(split='train', channels=3):
    path = f'data/usps-{split}.pkl'
    if not os.path.exists(path):
        dataset = USPS(f'{HOME}/.pytorch/USPS', train=(split=='train'), download=True)
        x, y = dataset.data, dataset.targets
        x = torch.tensor(resize_imgs(x, 32))
        x = (x.float()/255.).unsqueeze(1).repeat(1,3,1,1)
        y = torch.tensor(y)
        with open(path, 'wb') as f:
            pickle.dump([x, y], f)
    with open(path, 'rb') as f:
        x, y = pickle.load(f)
        if channels == 1:
            x = x[:,0:1,:,:]
    dataset = TensorDataset(x, y)
    return dataset

def load_svhn(split='train', channels=3):
    dataset = SVHN(f'{HOME}/.pytorch/SVHN', split=split, download=True)
    x, y = dataset.data, dataset.labels
    x = x.astype('float32')/255.
    x, y = torch.tensor(x), torch.tensor(y)
    if channels == 1:
        x = x.mean(1, keepdim=True)
    dataset = TensorDataset(x, y)
    return dataset

def load_syndigit(split='train', channels=3):
    path = f'data/synth_{split}_32x32.mat'
    data = loadmat(path)
    x, y = data['X'], data['y']
    x = np.transpose(x, [3, 2, 0, 1]).astype('float32')/255.
    y = y.squeeze()
    x, y = torch.tensor(x), torch.tensor(y)
    if channels == 1:
        x = x.mean(1, keepdim=True)
    dataset = TensorDataset(x, y)
    return dataset

def load_mnist_m(split='train', channels=3):
    path = f'data/mnist_m-{split}.pkl'
    with open(path, 'rb') as f:
        x, y = pickle.load(f)
        x, y = torch.tensor(x.astype('float32')/255.), torch.tensor(y)
        if channels==1:
            x = x.mean(1, keepdim=True)
    dataset = TensorDataset(x, y)
    return dataset

if __name__=='__main__':
    dataset = load_mnist(split='train')
    print('mnist train', len(dataset))
    dataset = load_mnist('test')
    print('mnist test', len(dataset))
    dataset = load_mnist_m('test')
    print('mnsit_m test', len(dataset))
    dataset = load_svhn(split='test')
    print('svhn', len(dataset))
    dataset = load_usps(split='test')
    print('usps', len(dataset))
    dataset = load_syndigit(split='test')
    print('syndigit', len(dataset))

