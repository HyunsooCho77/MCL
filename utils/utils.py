from torchvision.utils import save_image,make_grid
import numpy as np
import hydra
import torch
import torch.nn.functional as F


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_pth(epoch, model, optimizer, scheduler, fname):
    torch.save({
                'epoch': epoch,
                'net': model.module.state_dict(),
                'optim':optimizer.state_dict(),
                'scheduler':scheduler.state_dict()
    }, fname)


def load_pth(fname, model, optimizer = None, scheduler = None):
    # try :

    ckpt = torch.load(fname)
    resume_epoch = ckpt['epoch']
    model.load_state_dict(ckpt['net'])
    if optimizer != None :
        optimizer.load_state_dict(ckpt['optim'])
    if scheduler != None :
        scheduler.load_state_dict(ckpt['scheduler'])
    return resume_epoch



def print_lr(optimizer, print_screen=True,epoch = -1):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        if print_screen == True:
            print(f'learning rate : {lr:.3f}')
    return lr


def get_lr(step, total_steps, lr_max, lr_min):
    """Compute learning rate according to cosine annealing schedule."""
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))