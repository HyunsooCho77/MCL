import hydra
from omegaconf import DictConfig
import logging
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torchvision.datasets import CIFAR10,CIFAR100
from models.resnet import resnet18, resnet34, resnet50
import torch.nn as nn
from tqdm import tqdm
from models.models import *
from utils.utils import *
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import os
import pickle
from tqdm import tqdm
from evaluate.test_ood import Metric
from math import sqrt


def extract_train_reps(val_loader, model, args=None):
    if args.dataset =='cifar10':
        num_class = 10 
    
    # SimCLR training
    model.eval()
    val_bar = tqdm(val_loader)

    # To save time
    break_point = len(val_loader) // 5

    class_train_reps = [[] for i in range(num_class*4)]
    
    for idx,(img, label, rot) in enumerate(val_bar):
        img, label = img.cuda(), label.cuda()
        reps = model(img)[1]

        # Representation distribution
        reps = F.normalize(reps,dim=1)
        reps = reps.detach().cpu().numpy()
        
        # Keeps Label information if IND(to calculate recall)
        label = label.detach().cpu().numpy()

        for temp,(i1,i2,i3) in enumerate(zip(reps,label,rot)):
            class_train_reps[int(i2+i3*num_class)].append(i1)

        val_bar.set_description('Calculating MND from Training dataset ')
        
        if idx == break_point:
            break
        
    # list to numpy array
    class_train_reps = np.array(class_train_reps)

    return class_train_reps


def cal_MND_whole(args,train_reps):
    if args.dataset =='cifar10':
        num_class = 10 
    num_class = num_class *4 
    total = [[] for i in range(num_class)]
    for label in tqdm(range(num_class)):
        try: 
            # mean, covariance, inverse covariance (calculate Mahalanobis Distance)
            mean_np = np.mean(train_reps[label],axis = 0)
            cov_np = np.cov(train_reps[label], rowvar= False)
            cov = torch.from_numpy(cov_np)
            # too avoid Singular matrix 
            # add very small values to diagonal elements 
            m = 1e-9
            cov_np = cov_np + np.eye(cov_np.shape[1]) * m
            inv_cov_np = np.linalg.inv(cov_np)
            inv_cov = torch.from_numpy(inv_cov_np)
            
            total[label].append(mean_np)
            total[label].append(cov_np)
            total[label].append(inv_cov_np)
        except:
            pass

    return total


def mahalanobis(x, u, inverse_covariance):
    delta = x - u
    m = np.dot(np.dot(delta, inverse_covariance), delta)
    return np.sqrt(m)


def SEI(train_loader, test_ind_loader, test_ood_loader, epoch, model, args):
    train_reps = extract_train_reps(train_loader, model, args=args)
    MND = cal_MND_whole(args,train_reps)
    ind, acc = SEI_single(test_ind_loader, MND, epoch, model, args, isind = True)
    ood = SEI_single(test_ood_loader, MND, epoch, model, args, isind = False)

    # Calculate OOD performance
    metrics = Metric(ind= ind, ood=ood)
    with open('SEI performance_summary.txt','a') as f:
        f.write(f'epoch : {epoch}, SEI-method : {args.SEI_method}, SEI-way : {args.SEI_way}, acc : {acc}, FPR-95 : {metrics.fpr:.3f}, ALL AUROC : {metrics.aur:.3f}, AUPR-IN : {metrics.auprin:.3f}, AUPR-OUT : {metrics.auprout:.3f}\n')
    
    return metrics.aur


def SEI_single(test_loader, MND, epoch, model, args, MND1=None, isind = True):
    model.eval()
    
    if args.dataset =='cifar10':
        num_class = 10
    
    label_total, label_correct = 0,0
    
    mahal_list = []
    with torch.no_grad():
        pbar = tqdm(test_loader)
        
        #  one batch -> one data 
        for batch_idx, (imgs, cls_targets, rot_targets) in enumerate(pbar):
            
            mahal_sum = []
            imgs, cls_targets, rot_targets = imgs.cuda(), cls_targets.cuda(), rot_targets.cuda()
            
            projection = model(imgs)[1]
            feature = F.normalize(projection,dim=1)
            feature = feature.detach().cpu().numpy()

            # cal label
            pred = []
            for rot_idx, f in enumerate(feature):
                if args.SEI_way == 8 or args.SEI_way == 2:
                    if rot_idx % 2 == 0:
                        rot_idx = int(rot_idx/2)
                    else:
                        rot_idx = int((rot_idx-1)/2)
                
                mah_temp = []

                for l in range(num_class):
                    temp_label = rot_idx * num_class + l
                    mah_temp.append(mahalanobis(f, MND[temp_label][0], MND[temp_label][2]))
                pred.append(mah_temp)

            if args.SEI_method == 'min':
                for way in range(args.SEI_way):
                    min_mahaldist = 999999
                    final_cls_pred = -1
                    for pred_idx in range(args.SEI_way):
                        for j in range(num_class):
                            if pred[pred_idx][j] <= min_mahaldist:
                                min_mahaldist = pred[pred_idx][j]
                                final_cls_pred = j
                for i in range(args.SEI_way):
                    mahal_sum.append(pred[i][final_cls_pred])
        
                mahal_list.append(-1*min(mahal_sum))
            
            elif args.SEI_method == 'mean':
                if args.SEI_way == 2 :
                    final_cal = [sum(x) for x in zip(pred[0],pred[1])]
                elif args.SEI_way == 4 :
                    final_cal = [sum(x) for x in zip(pred[0],pred[1],pred[2],pred[3])]
                elif args.SEI_way == 8:
                    final_cal = [sum(x) for x in zip(pred[0],pred[1],pred[2],pred[3],pred[4],pred[5],pred[6],pred[7])]
                
                if args.SEI_way == 1 :
                    final_cls_pred = pred[0].index(min(pred[0]))
                else:
                    final_cls_pred = final_cal.index(min(final_cal))
                
                for i in range(args.SEI_way):
                    mahal_sum.append(pred[i][final_cls_pred])
                    mahal_sum_all = sum(mahal_sum)
                mahal_list.append(-1*mahal_sum_all)

            elif args.SEI_method == 'wa':
                temp_list = []
                for label in range(num_class):
                    num = []
                    denom = 0
                    final = 0
                    for sei in range(args.SEI_way):
                        sqrt_ratio = 1/pred[sei][label]
                        num.append(sqrt_ratio)
                        denom+=sqrt_ratio
                    
                    for i in range(args.SEI_way):
                        final += num[i]*pred[i][label]/denom
                    temp_list.append(final)
                final_cls_pred = temp_list.index(min(temp_list))
        
                mahal_list.append(-1*min(temp_list))

            score = -1*min(temp_list)

            if isind == True:
                if final_cls_pred == cls_targets[0]:
                    label_correct += 1
                    label_total += 1
                else:
                    label_total += 1
                
            if isind == True:
                label_acc = label_correct * 100/label_total
                pbar.set_description(f'{args.SEI_way}-way IND SEI epoch {epoch}, acc : {label_acc:.3f}')
            else:
                pbar.set_description(f'{args.SEI_way}-way OOD SEI epoch {epoch} ')
    
    if isind == True:
        return mahal_list, label_acc
    else:
        return mahal_list


if __name__ == '__main__':
    main()