
import torch
import numpy as np
from PIL import ImageColor

def label2color(X):
    ''' 将图像分割的label转换成color
        x ==> [N, w, h]
        input ==> pytorch 中的 tensor
        output ==> pytorch 中的 tensor
    '''
    id2rgb = [[0,0,0], [128,128,128], [128,0,0],[128,64,128], \
            [0,0,192],[64,64,128],[128,128,0],[192,192,128], \
            [64,0,128],[192,128,128],[64,64,0],[0,128,192], \
            [0,172,0],[0,128,128]]
    
    R = torch.zeros_like(X).float()
    G = torch.zeros_like(X).float()
    B = torch.zeros_like(X).float()

    for i,(r,g,b) in enumerate(id2rgb):
        r, b, g = r/255, b/255, g/255
        R[X==i] = r
        G[X==i] = g
        B[X==i] = b

    R = R.unsqueeze(1)
    G = G.unsqueeze(1)
    B = B.unsqueeze(1)

    rgb = torch.cat([R,G,B],1)
    return rgb

if __name__=='__main__':
    x = torch.ones(8, 128, 128)
    
    for i in range(8):
        for j in range(0,128,32):
            for k in range(0,128,32):
                idx = np.random.randint(0,13)
                x[i, j:j+32, k:k+32] = idx
    
    rgb = label2color(x)
    print(rgb.shape)
    print(rgb[0,:,0,0])





