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

class Get_gradient(nn.Module):
    def __init__(self):
        super(Get_gradient, self).__init__()
        kernel_v = [[0, -1, 0], 
                    [0, 0, 0], 
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0], 
                    [-1, 0, 1], 
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False).cuda()
        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False).cuda()

    def forward(self, x):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=1)

        x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding=1)
        x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding=1)

        x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v, padding=1)
        x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h, padding=1)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)

        x = torch.cat([x0, x1, x2], dim=1)
        return x

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

# reconstruction_function = nn.BCELoss()
reconstruction_function = nn.MSELoss()
# reconstruction_function.size_average = False
def loss_function(recon_x, x, mu, logvar):
    BCE = reconstruction_function(recon_x, x)

    # https://arxiv.org/abs/1312.6114 (Appendix B)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)

    return BCE + KLD


def test_finetune(net, data_path,device, bs, log_txt_file, val_datapath, models_train_best, models_train_last, lr=1e-4, lr_de_epoch=25000,
          epochs=100000, log_interval=100, val_interval=1000):
    
    loss = torch.nn.MSELoss()
    lossCE = torch.nn.CrossEntropyLoss()
    optimizer = Adam(net.parameters(), lr, weight_decay=1e-6)
    train_loader=DataLoader(VideoDataset(data_path,epochs*bs,use_flow=False), num_workers=4,
                              batch_size=bs, shuffle=False, drop_last=False,pin_memory=True)
    # loss = Loss().cuda()
    vggnet = VGG19()
    if torch.cuda.is_available():
        vggnet = torch.nn.DataParallel(vggnet).cuda()
    
    provider = LossProvider()
    loss_function4 = provider.get_loss_function('Watson-DFT', colorspace='RGB', pretrained=True, reduction='sum')
    loss_function4.to(device)


    
    # cri_gp = GradientPenaltyLoss(device=device).to(device)
    # WSloss1 = WSloss().to(device)
    best_p, best_j = 0, 1
    ave_loss, ave_m_loss, ave_c_loss, ave_s_loss, ave_i_loss = 0, 0, 0, 0, 0
    epoch=0
    xfm = DWTForward(J=3, mode='zero', wave='db3').to(device)  # Accepts all wave types available to PyWavelets
    ifm = DWTInverse(mode='zero', wave='db3').to(device)
    # net.eval()
    net.train()
    for data,mask in train_loader:
        epoch+=1
        
        data=data.view(-1,data.shape[2],data.shape[3],data.shape[4])
        mask=mask.view(-1,mask.shape[2],mask.shape[3])
        
        img,cls_gt,  mask_gt = data,torch.rand(bs,78),mask
         
        # net.zero_grad()
        img, cls_gt, mask_gt = img.cuda(), cls_gt.cuda(), mask_gt.cuda()
        # pred_cls, pred_mask = net(img)
        # all_loss, m_loss, c_loss, s_loss, iou_loss = loss(pred_mask.detach(), mask_gt, pred_cls, cls_gt)
        # pred_img, attn = net(img)
        with torch.no_grad():
            pred_img, resized_out, pred_imgr, mu, logvar,temp_out, i_non_zero_start, i_non_zero_end = net(img)
            # pred_img, resized_out, pred_imgr = net(img)

            

            cc = 255 * (((pred_img[2].clone()-pred_img[2].min())  / (pred_img[2].max().clone() - pred_img[2].min().clone()))).permute(1,2,0).cpu().detach().numpy()
            dd = 255 * (((pred_imgr[2].clone()- pred_imgr[2].min()) / (pred_imgr[2].max().clone() - pred_imgr[2].min().clone()))).permute(1,2,0).cpu().detach().numpy()
            cv2.imwrite("pred_img.jpg", cc)
            cv2.imwrite("pred_imgr.jpg", dd)
            imgmain = 255 * (((img[2].clone()-img[2].min())  / (img[2].max().clone() - img[2].min().clone()))).permute(1,2,0).cpu().detach().numpy()
            cv2.imwrite("img.jpg", imgmain)
            cv2.imwrite("attn.jpg", resized_out[2].cpu().detach().numpy()*255)
            # cv2.imwrite("resized_out.jpg", resized_out[0].permute(1,2,0).cpu().detach().numpy()*255)
            cv2.imwrite("resizedNormal.jpg", cv2.resize(imgmain, (180,224)))
            # img.requires_grad = True
            # get_grad = Get_gradient()
            pred_imgResized = 255 * (((pred_img[2].clone()-pred_img[2].min())  / (pred_img[2].max().clone() - pred_img[2].min().clone()))).permute(1,2,0).cpu().detach().numpy()

            # x22 = torch.zeros(5,3,224,112)
            # for kk in range(pred_img.shape[0]):
            #     x22[kk] = fn.resize(pred_img[kk], size=[224,112])
            cv2.imwrite("pred_imgResized.jpg", cv2.resize(pred_imgResized, (112,224)))


            # return x_input_final.to(self.device), aaa
            lossVAE = loss_function(pred_imgr, img, mu, logvar)
            # loss5 = loss_function(pred_imgr, img)
            lossMSE = loss(img,pred_imgr.to(device)) 
            # img = get_grad(img)
            # pred_imgr = get_grad(pred_imgr)
            # l_d_gp = 0.1 * cri_gp(img,pred_imgr.to(device)) 



            loss2 = 0
            x,y = img.clone(),pred_img.to(device)
            r=0.7 
            loss2 = 0 - ssim_loss(x, y).clone()
            l, m, h = 1, 1, 1
            for i in range(2):
                l, m, h = l * r * r, l * r * (1 - r), l * (1 - r) * (1 - r)
                x0, x1, x2 = dwt(x.clone())
                y0, y1, y2 = dwt(y.clone())
                loss2 = loss2.clone() - (ssim_loss(x1.clone(), y1.clone()) * 2 * m - ssim_loss(x2.clone(), y2.clone()) * h).clone()
                x, y = x0.clone(), y0.clone()
            lossSSIMDWT = loss2.clone() - (ssim_loss(x0.clone(), y0.clone()) * l).clone()

            lossVGG = perceptualLoss(img, pred_img, vggnet)
            lossVGG_part = 0
            for i_part in range(5):
                if i_non_zero_start[i_part] >5 and i_non_zero_end[i_part]<107:
                    aa=img[i_part,:,:,2*i_non_zero_start[i_part]-10: 2*i_non_zero_end[i_part]+10]
                    bb=pred_img[i_part,:,:,2*i_non_zero_start[i_part]-10: 2*i_non_zero_end[i_part]+10]
                else:
                    aa=img[i_part,:,:,2*i_non_zero_start[i_part]: 2*i_non_zero_end[i_part]]
                    bb=pred_img[i_part,:,:,2*i_non_zero_start[i_part]: 2*i_non_zero_end[i_part]]

                    # imgmain = 255 * (((aa.clone()-aa.min())  / (aa.max().clone() - aa.min().clone()))).permute(1,2,0).cpu().detach().numpy()
                # cv2.imwrite("img2.jpg", imgmain)
                lossVGG_part+= perceptualLoss(aa.unsqueeze(0), bb.unsqueeze(0), vggnet)
            lossVGG_part /=5
            

            Ylimg, Yhimg = xfm(img)
            Ylpred_img, Yhpred_img = xfm(pred_img)

            lossDWT = loss(Ylimg.to(device),Ylpred_img.to(device)) 
            # Y = ifm((Yl, Yh))


            # for ii in r
            lossWatsonDFT = loss_function4(img, pred_img)
            # img = img.to(device, dtype=torch.long)
            # pred_img = pred_img.to(device, dtype=torch.float)
            # img = img.to(device).long()
            # LCE = lossCE( pred_img,img)


            # mse_loss = nn.MSELoss(reduction='elementwise_mean')

            # all_loss = lossMSE.clone() + 0.01*lossVGG+ 0.01*abs(lossSSIMDWT.clone())  + 0.1*lossDWT + 0.01*lossVAE
            all_loss = 0.01*lossMSE.clone()  + 0.005*lossDWT+ 0.01*abs(lossSSIMDWT.clone())+ 0.0000001*lossWatsonDFT.clone() + 0.005*lossVGG + 0.005*lossVGG_part# + 0.0001*lossVAE + 0.01*lossVGG


            #all_loss.requires_grad = True
            # all_loss, m_loss, c_loss, s_loss, iou_loss = loss(pred_mask.detach(), mask_gt, pred_cls, cls_gt)
            try:
                # all_loss.backward()
                epoch_loss = all_loss.item()
                # m_l = m_loss.item()
                # c_l = c_loss.item()
                # s_l = s_loss.item()
                # i_l = iou_loss.item()
                ave_loss += epoch_loss
                # ave_m_loss += m_l
                # ave_c_loss += c_l
                # ave_s_loss += s_l
                # ave_i_loss += i_l
                # optimizer.step()

                if epoch % log_interval == 0:
                    
                    ave_loss1 = ave_loss / epoch
                    ave_m_loss = ave_m_loss / log_interval
                    ave_c_loss = ave_c_loss / log_interval
                    ave_s_loss = ave_s_loss / log_interval
                    ave_i_loss = ave_i_loss / log_interval
                    custom_print(datetime.datetime.now().strftime('%F %T') +
                                ' lr: %e, epoch: [%d/%d], all_loss: [%.4f], epoch_loss: [%.4f], c_loss: [%.4f], s_loss: [%.4f], i_loss: [%.4f]' %
                                (lr, epoch, epochs, ave_loss1, epoch_loss, ave_c_loss, ave_s_loss, ave_i_loss), log_txt_file, 'a+')
                    
                    
                    # ave_loss, ave_m_loss, ave_c_loss, ave_s_loss, ave_i_loss = 0, 0, 0, 0, 0
            except:
                pass
    torch.save(net.state_dict(), models_train_last)
