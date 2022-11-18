
from loss import Loss
from torch.optim import Adam, SGD
from tools import custom_print
import datetime
import torch
torch.autograd.set_detect_anomaly(True)
from val import validation,validation_with_flow
from torch.utils.data import DataLoader
from data_processed import VideoDataset
import cv2
import torchvision.transforms.functional as fn

from pytorch_wavelets import DWTForward, DWTInverse # (or import DWT, IDWT)
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.optim import SGD
import numpy
from einops import rearrange
import time
from transformer import Transformer
from Intra_MLP import index_points,knn_l2
import torchvision
import torchvision.transforms.functional as fn
from loss_grad import GradientPenaltyLoss

from PerceptualSimilarity.src.loss.loss_provider import LossProvider


import pytorch_ssim
import torch.nn as nn
from dwt import dwt

ssim_loss = pytorch_ssim.SSIM()

# vgg choice

import kornia

import numpy as np

from wavelet_ssim_loss import WSloss

def perceptualLoss(fakeIm, realIm, vggnet):
    '''
    use vgg19 conv1_2, conv2_2, conv3_3 feature, before relu layer
    '''

    weights = [1, 0.2, 0.04]
    features_fake = vggnet(fakeIm)
    features_real = vggnet(realIm)
    features_real_no_grad = [f_real.detach() for f_real in features_real]
    mse_loss = nn.MSELoss(reduction='elementwise_mean')

    loss = 0
    for i in range(len(features_real)):
        loss_i = mse_loss(features_fake[i], features_real_no_grad[i])
        loss = loss + loss_i * weights[i]

    return loss

class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        '''
         use vgg19 conv1_2, conv2_2, conv3_3 feature, before relu layer
        '''
        self.feature_list = [2, 7, 14]
        vgg19 = torchvision.models.vgg19(pretrained=True)

        self.model = torch.nn.Sequential(*list(vgg19.features.children())[:self.feature_list[-1]+1])


    def forward(self, x):
        x = (x-0.5)/0.5
        features = []
        for i, layer in enumerate(list(self.model)):
            x = layer(x)
            if i in self.feature_list:
                features.append(x)

        return features


img = cv2.imread("D:/retargeting/cvpr2023/figs/qualitative_results/2015/50p/5/img5.jpg")
img_retarget = cv2.imread("D:/retargeting/cvpr2023/figs/qualitative_results/2015/50p/5/MC5.jpg")
img_retarget = cv2.resize(img_retarget, (448,224))

img = torch.FloatTensor(img).permute(2,0,1)
img_retarget = torch.FloatTensor(img_retarget).permute(2,0,1)
vggnet = VGG19()
lossVGG_part= perceptualLoss(img.unsqueeze(0), img_retarget.unsqueeze(0), vggnet)
print(lossVGG_part/(224*448))
