import os
import torch
# from pycocotools import coco
import queue
import threading
from model_video import build_model, weights_init, MyEnsemble
from tools import custom_print
from train import train_finetune_with_flow,train_finetune
from test_finetune import  test_finetune
from val import validation
import time
import datetime
import collections
from torch.utils.data import DataLoader
import argparse
torch.backends.cudnn.benchmark = True

from model_video2 import build_model2

import cv2


if __name__ == '__main__':
    # train_val_config
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='./models/UFO_last_ft_3Days.pth',help="restore checkpoint")
    parser.add_argument('--use_flow',default=False, help="dataset for evaluation")
    parser.add_argument('--img_size',default=224, help="size of input image")
    parser.add_argument('--lr',default=1e-4, help="learning rate")
    parser.add_argument('--lr_de',default=20000, help="learning rate decay")
    parser.add_argument('--batch_size',default=1, help="batch size")
    parser.add_argument('--group_size',default=4, help="group size")
    parser.add_argument('--epochs',default=1000000, help="epoch")
    parser.add_argument('--train_datapath',default='DAVIS_FBMS_flow2/DAVIS_FBMS_flow/', help="training dataset")
    parser.add_argument('--val_datapath',default='DAVIS_FBMS_flow2/DAVIS_FBMS_flow/', help="training dataset")
    args = parser.parse_args()
    train_datapath = args.train_datapath

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
    net = build_model(device) #.to(device)
    net2 = build_model2(device) #.to(device)
    netEnsemble = MyEnsemble(net, net2) #.to(device)
    print(sum(p.numel() for p in netEnsemble.parameters() if p.requires_grad))

    for param in netEnsemble.parameters():
        param.requires_grad = True


    for p in net.sp1[0].parameters():
      p.requires_grad=False
    for p in net.sp2[0].parameters():
      p.requires_grad=False
    for p in net.cls[0].parameters():
      p.requires_grad=False
    for p in net.cls_m[0].parameters():
      p.requires_grad=False
    
    net=net.to(device)
    net=torch.nn.DataParallel(net)
    state_dict=torch.load(model_path, map_location=gpu_id)
    net.load_state_dict(state_dict)
    
    # net.train()
  


    net2=net2.to(device)
    net2=torch.nn.DataParallel(net2)  

    for param in net2.parameters():
        param.requires_grad = True

    net = MyEnsemble(net, net2)  

    for param in net.parameters():
        param.requires_grad = True

    # net=torch.nn.DataParallel(net)
    state_dict=torch.load('./models/UFO_last_ftTrained2Days.pth', map_location=gpu_id)
    net.load_state_dict(state_dict)


    net.to(device)
    net.eval()
    


    if use_flow==False:
        test_finetune(netEnsemble, train_datapath , device, batch_size, log_txt_file, val_datapath, models_train_best, models_train_last, lr, lr_de, epochs, log_interval, val_interval)
    else:
        train_finetune_with_flow(net, train_datapath , device, batch_size, log_txt_file, val_datapath, models_train_best, models_train_last, lr, lr_de, epochs, log_interval, val_interval)
