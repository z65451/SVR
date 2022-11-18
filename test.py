import os
import torch
# from pycocotools import coco
import queue
import threading
from model_video import build_model, weights_init, MyEnsemble
from tools import custom_print
from train import train_finetune_with_flow,train_finetune
from val import validation
import time
import datetime
import collections
from torch.utils.data import DataLoader
import argparse
torch.autograd.set_detect_anomaly(True)

torch.backends.cudnn.benchmark = True

from model_video2_seam import build_model2

from test_trained import test_finetune
import cv2

import sys
sys.path.append('NonUniformBlurKernelEstimation/')
if __name__ == '__main__':
    # train_val_config
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='./models/image_best.pth',help="restore checkpoint")
    parser.add_argument('--use_flow',default=False, help="dataset for evaluation")
    parser.add_argument('--img_size',default=224, help="size of input image")
    parser.add_argument('--lr',default=1e-3, help="learning rate")
    parser.add_argument('--lr_de',default=20000, help="learning rate decay")
    parser.add_argument('--batch_size',default=1, help="batch size")
    parser.add_argument('--group_size',default=4, help="group size")
    parser.add_argument('--epochs',default=10000000, help="epoch")
    # parser.add_argument('--train_datapath',default='DAVIS_FBMS_flow/DAVIS_FBMS_flow/', help="training dataset")
    parser.add_argument('--train_datapath_small',default='datasets/15/small/testing/', help="training dataset")
    parser.add_argument('--train_datapath_large',default='datasets/15/large/testning/', help="training dataset")
    parser.add_argument('--val_datapath',default='datasets/15/small/training/', help="training dataset")
    args = parser.parse_args()
    train_datapath_small = args.train_datapath_small
    train_datapath_large = args.train_datapath_large

    val_datapath = [args.val_datapath]
    
    # project config
    project_name = 'UFO'
    device = torch.device('cuda:0')
    img_size = args.img_size
    lr = args.lr
    lr_de = args.lr_de
    epochs = args.epochs
    batch_size = args.batch_size
    group_size = args.group_size
    log_interval = 1
    val_interval = 1000
    use_flow=args.use_flow
    if use_flow:
        from model_video_flow import build_model, weights_init
    # create log dir
    log_root = './logs'
    if not os.path.exists(log_root):
        os.makedirs(log_root)

    # create log txt
    log_txt_file = os.path.join(log_root, project_name + '_log.txt')
    custom_print(project_name, log_txt_file, 'w')

    # create model save dir
    models_root = './models'
    if not os.path.exists(models_root):
        os.makedirs(models_root)

    models_train_last = os.path.join(models_root, project_name + '_last_ft.pth')
    models_train_best = os.path.join(models_root, project_name + '_best_ft.pth')


    # continute load checkpoint
    model_path = args.model
    gpu_id='cuda:0'
    device = torch.device(gpu_id)
    net1 = build_model(device) #.to(device)
    net2 = build_model2(device) #.to(device)
    # netEnsemble = MyEnsemble(net, net2) #.to(device)
    cc=0
    # for param in netEnsemble.parameters():
    #     param.requires_grad = True
    #     cc+=1
    # print(sum(p.numel() for p in netEnsemble.parameters() if p.requires_grad))




    # for p in net.sp1[0].parameters():
    #   p.requires_grad=False
    # for p in net.sp2[0].parameters():
    #   p.requires_grad=False
    # for p in net.cls[0].parameters():
    #   p.requires_grad=False
    # for p in net.cls_m[0].parameters():
    #   p.requires_grad=False
    
    net1=net1.to(device)
    net1=torch.nn.DataParallel(net1)
    state_dict=torch.load(model_path, map_location=gpu_id)
    net1.load_state_dict(state_dict)
    
    # net.train()
  


    net2=net2.to(device)
    net2=torch.nn.DataParallel(net2)  
    print(sum(p.numel() for p in net2.parameters() if p.requires_grad))

    for param in net2.parameters():
        param.requires_grad = True

    for param in net1.parameters():
        param.requires_grad = False
    print(sum(p.numel() for p in net1.parameters() if p.requires_grad))

    net = MyEnsemble(net1, net2)  

    main_model_path = "./models/UFO_last_ftTrainedStereo.pth"
    # net=torch.nn.DataParallel(net)
    state_dict=torch.load(main_model_path, map_location=gpu_id)
    net.load_state_dict(state_dict)
    for param in net.parameters():
        param.requires_grad = False
    

    print(sum(p.numel() for p in net.parameters() if p.requires_grad))

    net.to(device)
    # net.eval()
    net.train()
    


    if use_flow==False:
        train_finetune(net, train_datapath_small, train_datapath_large , device, batch_size, log_txt_file, val_datapath, models_train_best, models_train_last, lr, lr_de, epochs, log_interval, val_interval)
    else:
        train_finetune_with_flow(net, train_datapath , device, batch_size, log_txt_file, val_datapath, models_train_best, models_train_last, lr, lr_de, epochs, log_interval, val_interval)
