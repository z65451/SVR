from loss import Loss
from torch.optim import Adam
from tools import custom_print
import datetime
import torch
from val import validation,validation_with_flow
from torch.utils.data import DataLoader
from data_processed import VideoDataset
import cv2
import torchvision.transforms.functional as fn


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
import torchvision.transforms.functional as fn

# vgg choice

import kornia

import numpy as np



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

    mask_pred_mid = torch.zeros(5,224,112)
    x_input_mid = torch.zeros(5,3,224,112)
    attn_input_mid = torch.zeros(5,1,224,112)
    x_input_final = torch.zeros(5,3,224,224)
    x_input_finalr = torch.zeros(5,3,224,224)
    x_mid_final = torch.zeros(5,3,224,112)

    for kk in range(x.shape[0]):
        # mask_pred_mid[kk] = fn.resize(mask_pred[kk].view(1,224,224), size=[112,224])
        x_input_mid[kk] = fn.resize(x[kk], size=[224,112])
        attn_input_mid[kk] = fn.resize(attn[kk], size=[224,112])
    

    x = x_input_mid.to(device)
    attn_input_mid2 = attn_input_mid.to(device)

    # attn_input_mid2 = attn_input_mid.to(self.device)+0.3 * attn_input_mid2
    # attn_input_mid2=torch.where(attn_input_mid2<float(attn_input_mid2.mean()), 0.0, 255.0)

    # bb = F.relu(self.conv1(attn_input_mid2))

    # cc = bb.tile((1,1,224,1))
    # cv2.imwrite("attn.jpg", attn[0].permute(1,2,0).cpu().detach().numpy()*255)
    # cv2.imwrite("attn2.jpg", cc[0].permute(1,2,0).cpu().detach().numpy()*255)

    # # cc=torch.where(cc<float(cc.mean()), 0.0, 255.0)
    # # cc = kornia.filters.box_blur(cc, (11,11), border_type='reflect', normalized=True)
    # cc = kornia.filters.gaussian_blur2d(cc, (11,11), sigma=(1,1))

    gray = (attn_input_mid2[0].permute(1,2,0).cpu().detach().numpy()*255).astype(np.uint8)*255
    img = gray 
    # thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY)[1]


    thresh = cv2.threshold(gray,4,255,0)[1]

    # apply morphology open to smooth the outline
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


        # draw the contour on a copy of the input image
        # results = img.copy()
        # #cv2.drawContours(results,[big_contour],0,(0,0,255),2)
        # x_counters = []
        # for count_er in big_contour:
        #     x=count_er[0][0]
        #     x_counters.append(x)

        # write result to disk
        # cv2.imwrite("greengreen_and_red_regions_threshold.png", thresh)
        # cv2.imwrite("green_and_red_regions_big_contour.png", results)


        # get contours
        # result = img.copy()
        # contours,hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # contours = contours[0] if len(contours) == 2 else contours[1]

        # sorted_contours= sorted(contours, key=cv2.contourArea, reverse= True)
        # largest_item= sorted_contours[0]

        # big_contour
        i_non_zero_start = min(x_counters)
        i_non_zero_end = max(x_counters)
        # save resulting image
        # cv2.imwrite('two_blobs_result.jpg',result*255)

        # noz = torch.nonzero(bb[0,0,0])

        # i_non_zero_start = float(noz[0])
        # i_non_zero_end = float(noz[-1])

        v0 = []
        for i in range(int(i_non_zero_start)):
            v0.append(i)

        v1 = []
        for i in range(int(i_non_zero_start), int(i_non_zero_end)):
            v1.append(i)

        v2 = []
        for i in range(int(i_non_zero_end), int(112)):
            v2.append(i)
        # cc = kornia.filters.box_blur(cc, (11,11), border_type='reflect', normalized=True)

        


        v0_F = False
        v1_F = False
        v2_F = False
        if len(v0) > 10:
            v0_F = True
            x0 = torch.zeros(5,3,224,len(v0)-int(10))
            x0r = torch.zeros(5,3,224,len(v0))
            for kk in range(x.shape[0]):
                # mask_pred_mid[kk] = fn.resize(mask_pred[kk].view(1,224,224), size=[112,224])
                x0[kk] = fn.resize(x_input_mid[kk,:,:,0:len(v0)], size=[224,len(v0)-int(10)])

        if len(v1) > 10:
            v1_F = True
            x1 = torch.zeros(5,3,224,len(v1)+int(20))
            x1r = torch.zeros(5,3,224,len(v1))
            for kk in range(x.shape[0]):
                # mask_pred_mid[kk] = fn.resize(mask_pred[kk].view(1,224,224), size=[112,224])
                x1[kk] = fn.resize(x_input_mid[kk,:,:,len(v0)+1:len(v0)+1+len(v1)], size=[224,len(v1)+int(20)])

        if len(v2) > 10:
            v2_F = True
            x2 = torch.zeros(5,3,224,len(v2)-int(10))
            x2r = torch.zeros(5,3,224,len(v2))
            for kk in range(x.shape[0]):
                # mask_pred_mid[kk] = fn.resize(mask_pred[kk].view(1,224,224), size=[112,224])
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
        aaa = torch.zeros(5,3,224,112)
        aaa2 = torch.zeros(5,3,224,112)

    
    # Do the reverse



    return x_input_final.to(device), aaa, x_input_finalr.to(device)



def test_finetune(net, data_path,device, bs, log_txt_file, val_datapath, models_train_best, models_train_last, lr=1e-4, lr_de_epoch=25000,
          epochs=100000, log_interval=100, val_interval=1000):
    
    loss = torch.nn.MSELoss()
    optimizer = Adam(net.parameters(), lr, weight_decay=1e-6)
    train_loader=DataLoader(VideoDataset(data_path,epochs*bs,use_flow=False), num_workers=4,
                              batch_size=bs, shuffle=True, drop_last=False,pin_memory=False)
    # loss = Loss().cuda()
    # torch.autograd.set_detect_anomaly(True)
    best_p, best_j = 0, 1
    ave_loss, ave_m_loss, ave_c_loss, ave_s_loss, ave_i_loss = 0, 0, 0, 0, 0
    epoch=0
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
            pred_img, resized_out, pred_imgr = net(img)

        

        #####################################################################3###


        # pred_img, resized_out, pred_imgr  = model22(pred_img, attn)


        ##########################################################################3
            # pred_img[2] = 255 * (pred_img[2] / (pred_img[2].max() - pred_img[2].min()))
            # cv2.imwrite("middleOutObjectBig.jpg", pred_img[2].permute(1,2,0).cpu().detach().numpy()*2550)
            cv2.imwrite("imgpredrObjectSmall.jpg", pred_imgr[2].permute(1,2,0).cpu().detach().numpy()*255)
            cv2.imwrite("img.jpg", img[2].permute(1,2,0).cpu().detach().numpy()*255)
            cv2.imwrite("attn.jpg", resized_out[2].cpu().detach().numpy()*255)
            # cv2.imwrite("resized_out.jpg", resized_out[0].permute(1,2,0).cpu().detach().numpy()*255)
            cv2.imwrite("img05ObjectSmall.jpg", fn.resize(img[2], size=[224,112]).permute(1,2,0).cpu().detach().numpy()*255)


        # # return x_input_final.to(self.device), aaa
        # all_loss = loss(img,pred_imgr.to(device)) 
        # #all_loss.requires_grad = True
        # # all_loss, m_loss, c_loss, s_loss, iou_loss = loss(pred_mask.detach(), mask_gt, pred_cls, cls_gt)
        # all_loss.backward()
        # epoch_loss = all_loss.item()
        # # m_l = m_loss.item()
        # # c_l = c_loss.item()
        # # s_l = s_loss.item()
        # # i_l = iou_loss.item()
        # ave_loss += epoch_loss
        # # ave_m_loss += m_l
        # # ave_c_loss += c_l
        # # ave_s_loss += s_l
        # # ave_i_loss += i_l
        # optimizer.step()

    #     if epoch % log_interval == 0:
            
    #         ave_loss = ave_loss / log_interval
    #         ave_m_loss = ave_m_loss / log_interval
    #         ave_c_loss = ave_c_loss / log_interval
    #         ave_s_loss = ave_s_loss / log_interval
    #         ave_i_loss = ave_i_loss / log_interval
    #         custom_print(datetime.datetime.now().strftime('%F %T') +
    #                      ' lr: %e, epoch: [%d/%d], all_loss: [%.4f], m_loss: [%.4f], c_loss: [%.4f], s_loss: [%.4f], i_loss: [%.4f]' %
    #                      (lr, epoch, epochs, ave_loss, ave_m_loss, ave_c_loss, ave_s_loss, ave_i_loss), log_txt_file, 'a+')
            
            
    #         ave_loss, ave_m_loss, ave_c_loss, ave_s_loss, ave_i_loss = 0, 0, 0, 0, 0
    # torch.save(net.state_dict(), models_train_last)
        # if epoch % val_interval == 0:
        #     net.eval()
        #     with torch.no_grad():
        #         custom_print(datetime.datetime.now().strftime('%F %T') +
        #                      ' now is evaluating the coseg dataset', log_txt_file, 'a+')
        #         ave_p, ave_j = validation(net, val_datapath, device, group_size=5, img_size=224, img_dir_name='image', gt_dir_name='groundtruth',
        #                                   img_ext=['.jpg', '.jpg', '.jpg', '.jpg'], gt_ext=['.png', '.bmp', '.jpg', '.png'])
        #         if ave_p[0] > best_p:
        #             # follow yourself save condition
        #             best_p = ave_p[0]
        #             best_j = ave_j[0]
        #             torch.save(net.state_dict(), models_train_best)
        #         torch.save(net.state_dict(), models_train_last)
                
        #         custom_print('-' * 100, log_txt_file, 'a+')
        #         custom_print(datetime.datetime.now().strftime('%F %T') + ' DAVIS  p: [%.4f], j: [%.4f]' %
        #                      (ave_p[0], ave_j[0]), log_txt_file, 'a+')
        #         custom_print('-' * 100, log_txt_file, 'a+')
        #     net.train()
            
        # if epoch % lr_de_epoch == 0:
        #     optimizer = Adam(net.parameters(), lr/2, weight_decay=1e-6)
        #     lr = lr / 2

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