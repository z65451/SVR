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
x_ratio = 224
# x_ratio = 360 #20p
# x_ratio = 320 # 30p
# x_ratio = 672




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

def train(net, device, q, log_txt_file, val_datapath, models_train_best, models_train_last, lr=1e-4, lr_de_epoch=25000,
          epochs=100000, log_interval=100, val_interval=1000):
    optimizer = SGD(net.parameters(), lr, weight_decay=1e-6)
    
    loss = Loss().cuda()
    best_p, best_j = 0, 1
    ave_loss, ave_m_loss, ave_c_loss, ave_s_loss, ave_i_loss = 0, 0, 0, 0, 0
    for epoch in range(1, epochs+1):
        img, cls_gt, mask_gt = q.get()
        net.zero_grad()
        img, cls_gt, mask_gt = img.cuda(), cls_gt.cuda(), mask_gt.cuda()
        
        pred_cls, pred_mask = net(img)
        all_loss, m_loss, c_loss, s_loss, iou_loss = loss(pred_mask, mask_gt, pred_cls, cls_gt)
        all_loss.backward()
        epoch_loss = all_loss.item()
        m_l = m_loss.item()
        c_l = c_loss.item()
        s_l = s_loss.item()
        i_l = iou_loss.item()
        ave_loss += epoch_loss
        ave_m_loss += m_l
        ave_c_loss += c_l
        ave_s_loss += s_l
        ave_i_loss += i_l
        optimizer.step()

        if epoch % log_interval == 0:
            ave_loss = ave_loss / log_interval
            ave_m_loss = ave_m_loss / log_interval
            ave_c_loss = ave_c_loss / log_interval
            ave_s_loss = ave_s_loss / log_interval
            ave_i_loss = ave_i_loss / log_interval
            custom_print(datetime.datetime.now().strftime('%F %T') +
                         ' lr: %e, epoch: [%d/%d], all_loss: [%.4f], m_loss: [%.4f], c_loss: [%.4f], s_loss: [%.4f], i_loss: [%.4f]' %
                         (lr, epoch, epochs, ave_loss, ave_m_loss, ave_c_loss, ave_s_loss, ave_i_loss), log_txt_file, 'a+')
            ave_loss, ave_m_loss, ave_c_loss, ave_s_loss, ave_i_loss = 0, 0, 0, 0, 0
  
        if epoch % val_interval == 0:
            net.eval()
            with torch.no_grad():
                custom_print(datetime.datetime.now().strftime('%F %T') +
                             ' now is evaluating the coseg dataset', log_txt_file, 'a+')
                ave_p, ave_j = validation(net, val_datapath, device, group_size=5, img_size=224, img_dir_name='image', gt_dir_name='groundtruth',
                                          img_ext=['.jpg', '.jpg', '.jpg', '.jpg'], gt_ext=['.png', '.bmp', '.jpg', '.png'])
                if ave_p[3] > best_p:
                    # follow yourself save condition
                    best_p = ave_p[3]
                    best_j = ave_j[0]
                    torch.save(net.state_dict(), models_train_best)
                torch.save(net.state_dict(), models_train_last)
                custom_print('-' * 100, log_txt_file, 'a+')
                custom_print(datetime.datetime.now().strftime('%F %T') + ' iCoseg8  p: [%.4f], j: [%.4f]' %
                             (ave_p[0], ave_j[0]), log_txt_file, 'a+')
                custom_print(datetime.datetime.now().strftime('%F %T') + ' MSRC7    p: [%.4f], j: [%.4f]' %
                             (ave_p[1], ave_j[1]), log_txt_file, 'a+')
                custom_print(datetime.datetime.now().strftime('%F %T') + ' Int_300  p: [%.4f], j: [%.4f]' %
                             (ave_p[2], ave_j[2]), log_txt_file, 'a+')
                custom_print(datetime.datetime.now().strftime('%F %T') + ' PAS_VOC  p: [%.4f], j: [%.4f]' %
                             (ave_p[3], ave_j[3]), log_txt_file, 'a+')
                custom_print('-' * 100, log_txt_file, 'a+')
            net.train()

        if epoch % lr_de_epoch == 0:
            optimizer = Adam(net.parameters(), lr/2, weight_decay=1e-6)
            lr = lr / 2


def rmodel22(x, attn):

    device = "cuda:0"



def model22(x, attn):

    device = "cuda:0"

    x_input = x
    
    attn = attn.view(5,1,224,224)
    attn = kornia.filters.gaussian_blur2d(attn, (11,11), sigma=(1,1))
    attn=torch.where(attn<float(attn.mean()), 0.0, 255.0)

    mask_pred_mid = torch.zeros(5,224,x_ratio)
    x_input_mid = torch.zeros(5,3,224,x_ratio)
    attn_input_mid = torch.zeros(5,1,224,x_ratio)
    x_input_final = torch.zeros(5,3,224,224)
    x_input_finalr = torch.zeros(5,3,224,224)
    x_mid_final = torch.zeros(5,3,224,x_ratio)

    for kk in range(x.shape[0]):
        # mask_pred_mid[kk] = fn.resize(mask_pred[kk].view(1,224,224), size=[112,224])
        x_input_mid[kk] = fn.resize(x[kk], size=[224,x_ratio])
        attn_input_mid[kk] = fn.resize(attn[kk], size=[224,x_ratio])
    

    x = x_input_mid.to(device)
    attn_input_mid2 = attn_input_mid.to(device)


    gray = (attn_input_mid2[0].permute(1,2,0).cpu().detach().numpy()*255).astype(np.uint8)*255
    img = gray 


    thresh = cv2.threshold(gray,4,255,0)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # find contours
    cntrs = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]

    # Contour filtering to get largest area
    area_thresh = 0
    for c in cntrs:
        area = cv2.contourArea(c)
        if area > area_thresh:
            area = area_thresh
            big_contour = c
    aa=[]
    for i in range(len(cntrs)):
        aa=np.append(aa,cntrs[i])
    
    if len(aa ) != 0:
        x_counters = []
        for jj in range(len(aa)):
            if jj%2==0:
                x_counters.append(int(aa[jj]))



        # big_contour
        i_non_zero_start = min(x_counters)
        i_non_zero_end = max(x_counters)

        v0 = []
        for i in range(int(i_non_zero_start)):
            v0.append(i)

        v1 = []
        for i in range(int(i_non_zero_start), int(i_non_zero_end)):
            v1.append(i)

        v2 = []
        for i in range(int(i_non_zero_end), int(x_ratio)):
            v2.append(i)

        v0_F = False
        v1_F = False
        v2_F = False
        if len(v0) > 10:
            v0_F = True
            x0 = torch.zeros(5,3,224,len(v0)-int(10))
            x0r = torch.zeros(5,3,224,len(v0))
            for kk in range(x.shape[0]):
                x0[kk] = fn.resize(x_input_mid[kk,:,:,0:len(v0)], size=[224,len(v0)-int(10)])

        if len(v1) > 10:
            v1_F = True
            x1 = torch.zeros(5,3,224,len(v1)+int(20))
            x1r = torch.zeros(5,3,224,len(v1))
            for kk in range(x.shape[0]):
                x1[kk] = fn.resize(x_input_mid[kk,:,:,len(v0)+1:len(v0)+1+len(v1)], size=[224,len(v1)+int(20)])

        if len(v2) > 10:
            v2_F = True
            x2 = torch.zeros(5,3,224,len(v2)-int(10))
            x2r = torch.zeros(5,3,224,len(v2))
            for kk in range(x.shape[0]):
                x2[kk] = fn.resize(x_input_mid[kk,:,:,len(v0)+1+len(v1):len(v0)+1+len(v1)+len(v2)], size=[224,len(v2)-int(10)])
        if v0_F and v1_F and v2_F:
            ff=torch.cat((x0,x1,x2), dim=-1)

            for kk in range(x.shape[0]):
                x0r[kk] = fn.resize(x0[kk,:,:,:], size=[224,len(v0)])

            for kk in range(x.shape[0]):
                x1r[kk] = fn.resize(x1[kk,:,:,:], size=[224,len(v1)])
            for kk in range(x.shape[0]):
                x2r[kk] = fn.resize(x2[kk,:,:,:], size=[224,len(v2)])


            ffr = torch.cat((x0r,x1r,x2r), dim=-1)

            cv2.imwrite("retarg.jpg", ff[0].permute(1,2,0).cpu().detach().numpy()*255)
            cv2.imwrite("retargBack.jpg", ffr[0].permute(1,2,0).cpu().detach().numpy()*255)
        else:
            if v1_F and v2_F:
                ff=torch.cat((x1,x2), dim=-1)

                for kk in range(x.shape[0]):
                    x1r[kk] = fn.resize(x1[kk,:,:,:], size=[224,len(v1)])
                for kk in range(x.shape[0]):
                    x2r[kk] = fn.resize(x2[kk,:,:,:], size=[224,len(v2)])


                ffr = torch.cat((x1r,x2r), dim=-1)

            else:
                if v0_F and v1_F:
                    ff=torch.cat((x0,x1), dim=-1)
                    for kk in range(x.shape[0]):
                        x0r[kk] = fn.resize(x0[kk,:,:,:], size=[224,len(v0)])
                    for kk in range(x.shape[0]):
                        x1r[kk] = fn.resize(x1[kk,:,:,:], size=[224,len(v1)])

                    ffr = torch.cat((x0r,x1r), dim=-1)

                if v0_F and v2_F:
                    ff=torch.cat((x0,x2), dim=-1)
                    for kk in range(x.shape[0]):
                        x0r[kk] = fn.resize(x0[kk,:,:,:], size=[224,len(v0)])
                    for kk in range(x.shape[0]):
                        x2r[kk] = fn.resize(x2[kk,:,:,:], size=[224,len(v2)])

                    ffr = torch.cat((x0r,x2r), dim=-1)

                if v1_F and not v2_F and not v0_F:
                    ff=x1
                    for kk in range(x.shape[0]):
                        x1r[kk] = fn.resize(x1[kk,:,:,:], size=[224,len(v1)])

                    ffr = x1r 

        aaa = ff
        aaar = ffr

        for kk in range(x.shape[0]):
            x_input_final[kk] = fn.resize(aaa[kk], size=[224,224])
            x_input_finalr[kk] = fn.resize(aaar[kk], size=[224,224])
    else:
        aaa = torch.zeros(5,3,224,x_ratio)
        aaa2 = torch.zeros(5,3,224,x_ratio)


    return x_input_final.to(device), aaa, x_input_finalr.to(device)

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


def train_finetune(train_or_test,net, train_datapath_small, train_datapath_large,device, bs, log_txt_file, val_datapath, models_train_best, models_train_last, lr=1e-4, lr_de_epoch=25000,
          epochs=100000, log_interval=100, val_interval=1000):
    
    loss = torch.nn.MSELoss()
    lossCE = torch.nn.CrossEntropyLoss()
    optimizer = Adam(net.parameters(), lr, weight_decay=1e-6)
    train_loader=DataLoader(VideoDataset(train_or_test, train_datapath_small, train_datapath_large,epochs*bs,use_flow=False), num_workers=0,
                              batch_size=bs, shuffle=True, drop_last=False,pin_memory=True)
    # loss = Loss().cuda()
    vggnet = VGG19()
    if torch.cuda.is_available():
        vggnet = torch.nn.DataParallel(vggnet).cuda()
    
    provider = LossProvider()
    loss_function4 = provider.get_loss_function('Watson-DFT', colorspace='RGB', pretrained=True, reduction='sum')
    loss_function4.to(device)
    criterion_L1 = torch.nn.L1Loss().to(device)


    
    # cri_gp = GradientPenaltyLoss(device=device).to(device)
    # WSloss1 = WSloss().to(device)
    best_p, best_j = 0, 1
    ave_loss, ave_m_loss, ave_c_loss, ave_s_loss, ave_i_loss = 0, 0, 0, 0, 0
    epoch=0
    xfm = DWTForward(J=3, mode='zero', wave='db3').to(device)  # Accepts all wave types available to PyWavelets
    ifm = DWTInverse(mode='zero', wave='db3').to(device)
    net.train()
    # if train_or_test=="train":
    #     net.train()
    # else:
    #     net.eval()
    for data_l,data_r, disp_l, disp_r, tmp_img_l_o, tmp_img_r_o in train_loader:
        epoch+=1
        
        data_l=data_l.view(-1,data_l.shape[2],data_l.shape[3],data_l.shape[4])
        data_r=data_r.view(-1,data_r.shape[2],data_r.shape[3],data_r.shape[4])
        disp_l=disp_l.view(-1,disp_l.shape[2],disp_l.shape[3],disp_l.shape[4])
        disp_r=disp_r.view(-1,disp_r.shape[2],disp_r.shape[3],disp_r.shape[4])
        # mask=mask.view(-1,mask.shape[2],mask.shape[3])
        
        img_l = data_l
        img_r = data_r
        disp_l = disp_l
        disp_r = disp_r
         
        net.zero_grad()
        img_l = img_l.cuda()
        img_r = img_r.cuda()
        disp_l = disp_l.cuda()
        disp_r = disp_r.cuda()
        imgmain = 255 * (((img_l[2].clone()-img_l[2].min())  / (img_l[2].max().clone() - img_l[2].min().clone()))).permute(1,2,0).cpu().detach().numpy()
        imgmain_r = 255 * (((img_r[2].clone()-img_r[2].min())  / (img_r[2].max().clone() - img_r[2].min().clone()))).permute(1,2,0).cpu().detach().numpy()
        cv2.imwrite("imgl.jpg", imgmain)
        cv2.imwrite("imgr.jpg", imgmain_r)
        cv2.imwrite("resizedNormal_l.jpg", cv2.resize(imgmain, (x_ratio,224)))
        cv2.imwrite("resizedNormal_r.jpg", cv2.resize(imgmain_r, (x_ratio,224)))

        pred_img, resized_out, pred_imgr,temp_out, i_non_zero_start, i_non_zero_end, pred_img_r, resized_out_r, pred_imgr_r,temp_out_r, i_non_zero_start_r, i_non_zero_end_r, img_l_out, img_r_out, M_right_to_left, M_left_to_right, V_left, V_right = net(img_l, img_r, disp_l, disp_r,tmp_img_l_o, tmp_img_r_o)
        
        img_l_resized = fn.resize(img_l, size=[224, x_ratio])
        img_r_resized = fn.resize(img_r, size=[224, x_ratio])
        img_l_out = img_l_out.to(device)
        img_r_out = img_r_out.to(device)
        
        ''' Smoothness Loss '''
        loss_h = criterion_L1(M_right_to_left[:-1, :, :], M_right_to_left[:1:, :, :]) + \
                 criterion_L1(M_left_to_right[:-1, :, :], M_left_to_right[1:, :, :])
        loss_w = criterion_L1(M_right_to_left[:, :-1, :-1], M_right_to_left[:, 1:, 1:]) + \
                 criterion_L1(M_left_to_right[:, :-1, :-1], M_left_to_right[:, 1:, 1:])
        loss_smooth = loss_w + loss_h

        ''' Consistency Loss '''
        b=5
        c=3
        w=x_ratio
        h=224
        SR_left_res = F.interpolate(torch.abs(img_l_resized - img_l_out.permute(0,3,1,2)), scale_factor=1 / 1, mode='bicubic', align_corners=False)
        SR_right_res = F.interpolate(torch.abs(img_r_resized - img_r_out.permute(0,3,1,2)), scale_factor=1 / 1, mode='bicubic', align_corners=False)
        SR_left_resT = torch.bmm(M_right_to_left.detach().contiguous().view(b * h, w, w), SR_right_res.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                 ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
        SR_right_resT = torch.bmm(M_left_to_right.detach().contiguous().view(b * h, w, w), SR_left_res.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                  ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
        loss_cons = criterion_L1(SR_left_res * V_left.repeat(1, 3, 1, 1), SR_left_resT * V_left.repeat(1, 3, 1, 1)) + \
                   criterion_L1(SR_right_res * V_right.repeat(1, 3, 1, 1), SR_right_resT * V_right.repeat(1, 3, 1, 1))


        #####################################################################3###
        cc = 255 * (((pred_img[2].clone()-pred_img[2].min())  / (pred_img[2].max().clone() - pred_img[2].min().clone()))).permute(1,2,0).cpu().detach().numpy()
        dd = 255 * (((pred_imgr[2].clone()- pred_imgr[2].min()) / (pred_imgr[2].max().clone() - pred_imgr[2].min().clone()))).permute(1,2,0).cpu().detach().numpy()
        cv2.imwrite("pred_img.jpg", cc)
        cv2.imwrite("pred_imgr.jpg", dd)
        imgmain = 255 * (((img_l[2].clone()-img_l[2].min())  / (img_l[2].max().clone() - img_l[2].min().clone()))).permute(1,2,0).cpu().detach().numpy()
        imgmain_r = 255 * (((img_r[2].clone()-img_r[2].min())  / (img_r[2].max().clone() - img_r[2].min().clone()))).permute(1,2,0).cpu().detach().numpy()
        cv2.imwrite("imgl.jpg", imgmain)
        cv2.imwrite("imgr.jpg", imgmain_r)
        cv2.imwrite("attn_l.jpg", resized_out[2].cpu().detach().numpy()*255)
        cv2.imwrite("attn_r.jpg", resized_out_r[2].cpu().detach().numpy()*255)
        cv2.imwrite("resizedNormal_l.jpg", cv2.resize(imgmain, (x_ratio,224)))
        cv2.imwrite("resizedNormal_r.jpg", cv2.resize(imgmain_r, (x_ratio,224)))
        pred_imgResized = 255 * (((pred_img[2].clone()-pred_img[2].min())  / (pred_img[2].max().clone() - pred_img[2].min().clone()))).permute(1,2,0).cpu().detach().numpy()

        cv2.imwrite("pred_imgResized.jpg", cv2.resize(pred_imgResized, (x_ratio,224)))

        lossMSE = loss(img_l,pred_imgr.to(device)) 
        lossMSE_r = loss(img_r,pred_imgr_r.to(device)) 

        loss2 = 0
        loss2_r = 0
        x,y = img_l.clone(),pred_imgr.to(device)
        x_r,y_r = img_r.clone(),pred_imgr_r.to(device)
        r=0.7 
        loss2 = 0 - ssim_loss(x, y).clone()
        loss2_r = 0 - ssim_loss(x_r, y_r).clone()
        l, m, h = 1, 1, 1


        for i in range(2):
            l, m, h = l * r * r, l * r * (1 - r), l * (1 - r) * (1 - r)
            x0, x1, x2 = dwt(x.clone())
            y0, y1, y2 = dwt(y.clone())
            x0_r, x1_r, x2_r = dwt(x_r.clone())
            y0_r, y1_r, y2_r = dwt(y_r.clone())
            loss2 = loss2.clone() - (ssim_loss(x1.clone(), y1.clone()) * 2 * m - ssim_loss(x2.clone(), y2.clone()) * h).clone()
            loss2_r = loss2_r.clone() - (ssim_loss(x1_r.clone(), y1_r.clone()) * 2 * m - ssim_loss(x2_r.clone(), y2_r.clone()) * h).clone()
            x, y = x0.clone(), y0.clone()
            x_r, y_r = x0_r.clone(), y0_r.clone()
        lossSSIMDWT = loss2.clone() - (ssim_loss(x0.clone(), y0.clone()) * l).clone()
        lossSSIMDWT_r = loss2_r.clone() - (ssim_loss(x0_r.clone(), y0_r.clone()) * l).clone()
        try:
            lossVGG = perceptualLoss(img_l_resized , img_l_out.permute(0,3,1,2), vggnet)
            lossVGG_r = perceptualLoss(img_r_resized , img_r_out.permute(0,3,1,2), vggnet)
            lossVGG_part = 0
            lossVGG_part_r = 0 
            for i_part in range(5):
                if i_non_zero_start[i_part] >5 and i_non_zero_end[i_part]<107:
                    aa=img_l[i_part,:,:,2*i_non_zero_start[i_part]-10: 2*i_non_zero_end[i_part]+10]
                    bb=pred_imgr[i_part,:,:,2*i_non_zero_start[i_part]-10: 2*i_non_zero_end[i_part]+10]
                else:
                    aa=img_l[i_part,:,:,2*i_non_zero_start[i_part]: 2*i_non_zero_end[i_part]]
                    bb=pred_imgr[i_part,:,:,2*i_non_zero_start[i_part]: 2*i_non_zero_end[i_part]]
            for i_part_r in range(5):
                if i_non_zero_start_r[i_part_r] >5 and i_non_zero_end_r[i_part_r]<107:
                    aa_r=img_r[i_part_r,:,:,2*i_non_zero_start_r[i_part_r]-10: 2*i_non_zero_end_r[i_part_r]+10]
                    bb_r=pred_imgr_r[i_part_r,:,:,2*i_non_zero_start_r[i_part_r]-10: 2*i_non_zero_end_r[i_part_r]+10]
                else:
                    aa_r=img_r[i_part_r,:,:,2*i_non_zero_start_r[i_part_r]: 2*i_non_zero_end_r[i_part_r]]
                    bb_r=pred_imgr_r[i_part_r,:,:,2*i_non_zero_start_r[i_part_r]: 2*i_non_zero_end_r[i_part_r]]

                lossVGG_part+= perceptualLoss(aa.unsqueeze(0), bb.unsqueeze(0), vggnet)
                lossVGG_part_r+= perceptualLoss(aa_r.unsqueeze(0), bb_r.unsqueeze(0), vggnet)
            lossVGG_part /=5
            lossVGG_part_r /=5
        except:
            lossVGG_part = 0
            lossVGG_part_r = 0
            lossVGG = 0
            lossVGG_r = 0
            pass


        Ylimg, Yhimg = xfm(img_l_resized)
        Ylpred_img, Yhpred_img = xfm(img_l_out.permute(0,3,1,2))
        lossDWT = loss(Ylimg.to(device),Ylpred_img.to(device)) 
    

        Ylimg_r, Yhimg = xfm(img_r_resized)
        Ylpred_img_r, Yhpred_img = xfm(img_r_out.permute(0,3,1,2))
        lossDWT_r = loss(Ylimg_r.to(device),Ylpred_img_r.to(device)) 

        lossWatsonDFT = loss_function4(img_l_resized, img_r_out.permute(0,3,1,2))
        lossWatsonDFT_r = loss_function4(img_r_resized, img_r_out.permute(0,3,1,2))
        all_loss_l = 0.001*lossMSE.clone()  + 0.05*lossDWT+ 0.001*abs(lossSSIMDWT.clone())+ 0.000001*lossWatsonDFT.clone() + 0.05*lossVGG + 0.0005*lossVGG_part# + 0.0001*lossVAE + 0.01*lossVGG
        all_loss_r = 0.001*lossMSE_r.clone()  + 0.05*lossDWT_r+ 0.001*abs(lossSSIMDWT_r.clone())+ 0.000001*lossWatsonDFT_r.clone() + 0.05*lossVGG_r + 0.0005*lossVGG_part_r# + 0.0001*lossVAE + 0.01*lossVGG
        all_loss = all_loss_l + all_loss_r + loss_smooth + loss_cons

        try:
            all_loss.backward()
            epoch_loss = all_loss.item()
            ave_loss += epoch_loss
            optimizer.step()

            if epoch % log_interval == 0:
                
                ave_loss1 = ave_loss / epoch
                ave_m_loss = ave_m_loss / log_interval
                ave_c_loss = ave_c_loss / log_interval
                ave_s_loss = ave_s_loss / log_interval
                ave_i_loss = ave_i_loss / log_interval
                custom_print(datetime.datetime.now().strftime('%F %T') +
                            ' lr: %e, epoch: [%d/%d], all_loss: [%.4f], epoch_loss: [%.4f], c_loss: [%.4f], s_loss: [%.4f], i_loss: [%.4f]' %
                            (lr, epoch, epochs, ave_loss1, epoch_loss, ave_c_loss, ave_s_loss, ave_i_loss), log_txt_file, 'a+')
                
                
        except:
            pass
    torch.save(net.state_dict(), models_train_last)

def train_finetune_with_flow(net, data_path,device, bs, log_txt_file, val_datapath, models_train_best, models_train_last, lr=1e-4, lr_de_epoch=25000,
          epochs=100000, log_interval=100, val_interval=1000):
    optimizer = Adam(net.parameters(), lr, weight_decay=1e-6)
    train_loader=DataLoader(VideoDataset(data_path,epochs*bs,use_flow=True), num_workers=4,
                              batch_size=bs, shuffle=True, drop_last=False,pin_memory=False)
    loss = Loss().cuda()
    best_p, best_j = 0, 1
    ave_loss, ave_m_loss, ave_c_loss, ave_s_loss, ave_i_loss = 0, 0, 0, 0, 0
    epoch=0
    for data,flow,mask in train_loader:
        epoch+=1
        
        data=data.view(-1,data.shape[2],data.shape[3],data.shape[4])
        flow=flow.view(-1,flow.shape[2],flow.shape[3],flow.shape[4])
        mask=mask.view(-1,mask.shape[2],mask.shape[3])
        flow=flow.cuda()
        img,cls_gt,  mask_gt = data,torch.rand(bs,78),mask
        
        
        net.zero_grad()
        img, cls_gt, mask_gt = img.cuda(), cls_gt.cuda(), mask_gt.cuda()
        pred_cls,pred_mask = net(img,flow)
        all_loss, m_loss, c_loss, s_loss, iou_loss = loss(pred_mask, mask_gt, pred_cls, cls_gt)
        all_loss.backward()
        epoch_loss = all_loss.item()
        m_l = m_loss.item()
        c_l = c_loss.item()
        s_l = s_loss.item()
        i_l = iou_loss.item()
        ave_loss += epoch_loss
        ave_m_loss += m_l
        ave_c_loss += c_l
        ave_s_loss += s_l
        ave_i_loss += i_l
        optimizer.step()

        if epoch % log_interval == 0:
            
            ave_loss = ave_loss / log_interval
            ave_m_loss = ave_m_loss / log_interval
            ave_c_loss = ave_c_loss / log_interval
            ave_s_loss = ave_s_loss / log_interval
            ave_i_loss = ave_i_loss / log_interval
            custom_print(datetime.datetime.now().strftime('%F %T') +
                         ' lr: %e, epoch: [%d/%d], all_loss: [%.4f], m_loss: [%.4f], c_loss: [%.4f], s_loss: [%.4f], i_loss: [%.4f]' %
                         (lr, epoch, epochs, ave_loss, ave_m_loss, ave_c_loss, ave_s_loss, ave_i_loss), log_txt_file, 'a+')
            
            ave_loss, ave_m_loss, ave_c_loss, ave_s_loss, ave_i_loss = 0, 0, 0, 0, 0
        
        if epoch % val_interval == 0:
            net.eval()
            with torch.no_grad():
                custom_print(datetime.datetime.now().strftime('%F %T') +
                             ' now is evaluating the coseg dataset', log_txt_file, 'a+')
                ave_p, ave_j = validation_with_flow(net, val_datapath, device, group_size=5, img_size=224, img_dir_name='image', gt_dir_name='groundtruth',
                                          img_ext=['.jpg', '.jpg', '.jpg', '.jpg'], gt_ext=['.png', '.bmp', '.jpg', '.png'])
                
                if ave_p[0] > best_p:
                    # follow yourself save condition
                    best_p = ave_p[0]
                    best_j = ave_j[0]
                    torch.save(net.state_dict(), models_train_best)
                torch.save(net.state_dict(), models_train_last)

                custom_print('-' * 100, log_txt_file, 'a+')
                custom_print(datetime.datetime.now().strftime('%F %T') + ' DAVIS  p: [%.4f], j: [%.4f]' %
                             (ave_p[0], ave_j[0]), log_txt_file, 'a+')
               
                custom_print('-' * 100, log_txt_file, 'a+')
            net.train()
        
        if epoch % lr_de_epoch == 0:
            optimizer = Adam(net.parameters(), lr/2, weight_decay=1e-6)
            lr = lr / 2