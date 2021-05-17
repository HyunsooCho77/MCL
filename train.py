import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from models.models import *
from utils.data_loader import *
from utils.loss import *
from utils.utils import *


def train_MCL(train_loader, model, optimizer, scheduler,  epoch, args):
    model.train()
    loss_meter = AverageMeter("MCL_loss")
    train_bar = tqdm(train_loader)
    lr = print_lr(optimizer, epoch)
    total_mcl_loss, total_CCM_loss, total_SPA_loss = 0, 0, 0
    
    if args.dataset == 'cifar10':
        num_class = 10

    for idx, (img, target, rot_target) in enumerate(train_bar):
        
        # batch * 2(pair) * channel *width * height
        sizes = img.size()
        img = img.view(sizes[0] * 2, sizes[2], sizes[3], sizes[4]).cuda(non_blocking=True)
        
        optimizer.zero_grad()
        rep= model(img)[1]
        
        # class conditional mask
        mcl_target = target + rot_target * num_class
        CCM_loss = nt_xent_CCM(rep, mcl_target, beta = args.beta, alpha = args.alpha, t = args.temperature)
        #  if args.CCM == True else nt_xent(rep, args.temperature)
        total_CCM_loss += CCM_loss.item()
        total_loss = CCM_loss

        # positive attraction mask
        try:
            if args.CCM == True and args.SPA == True :
                SPA_loss = args.SPA_weight * nt_xent_SPA(rep, mcl_target, beta = args.beta, alpha = args.alpha, t=  args.temperature)
                total_SPA_loss += SPA_loss.item()
                total_loss += SPA_loss

                train_bar.set_description(f"Train epoch {epoch}, lr : {lr:.4f}, Total loss : {(total_CCM_loss + total_SPA_loss)/(idx+1):.3f}, CCM loss: {total_CCM_loss/(idx+1):.3f}, SPA loss: {total_SPA_loss/(idx+1):.3f}")
            else:
                train_bar.set_description(f"Train epoch {epoch}, lr : {lr:.4f}, Loss: {total_CCM_loss/(idx+1):.3f}")
        except:
            print('exception raised')
            pass

        total_loss.backward()
        optimizer.step()
        scheduler.step()
           
    with open('train.log','a') as f:
        f.write(f"Epoch {epoch}, LR : {lr:.3f}, Total loss : {(total_CCM_loss + total_SPA_loss)/(idx+1):.3f}, CCM loss: {total_CCM_loss/(idx+1):.3f}, SPA loss: {total_SPA_loss/(idx+1):.3f}, \n")

