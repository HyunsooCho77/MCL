import os
import torch
import torchvision
from random import shuffle
import pickle as p
import numpy as np
from PIL import Image
from torchvision.utils import save_image,make_grid
from torchvision.datasets import CIFAR10, CIFAR100,SVHN
import argparse
import json
import torchvision.transforms.functional as TF
import random
import os


def data_download_preprocess(dset):
    if dset == 'svhn': 
        d_train = torchvision.datasets.SVHN('data/svhn', split='train', transform=None, target_transform=None, download=True)
        d_test = torchvision.datasets.SVHN('data/svhn', split='test', transform=None, target_transform=None, download=True)

        p_train, p_test = [], []
        
        for img,target in zip(d_train.data,d_train.labels):
            l = [img.transpose(1,2,0),int(target)]
            p_train.append(l)
            
        
        for img,target in zip(d_test.data, d_test.labels):
            l = [img.transpose(1,2,0),int(target)]
            p_test.append(l)
        
        p.dump(p_train,open('data/svhn/train_combined.pkl','wb'))
        p.dump(p_test,open('data/svhn/test_combined.pkl','wb'))
    
    
    elif dset =='cifar10':
        d_train = torchvision.datasets.CIFAR10('data/cifar10', train=True, transform=None, target_transform=None, download=True)
        d_test = torchvision.datasets.CIFAR10('data/cifar10', train=False, transform=None, target_transform=None, download=True)

        p_train, p_test = [], []

        for img,target in zip(d_train.data,d_train.targets):
            l = [img,int(target)]
            p_train.append(l)

        for img,target in zip(d_test.data, d_test.targets):
            l = [img,int(target)]
            p_test.append(l)
        
        p.dump(p_train,open('data/cifar10/train_combined.pkl','wb'))
        p.dump(p_test,open('data/cifar10/test_combined.pkl','wb'))

    elif dset =='cifar100':
        d_train = torchvision.datasets.CIFAR100('data/cifar100', train=True, transform=None, target_transform=None, download=True)
        d_test = torchvision.datasets.CIFAR100('data/cifar100', train=False, transform=None, target_transform=None, download=True)

        p_train, p_test = [], []
        for img,target in zip(d_train.data,d_train.targets):
            l = [img,int(target)]
            p_train.append(l)

        for img,target in zip(d_test.data, d_test.targets):
            l = [img,int(target)]
            p_test.append(l)
        
        p.dump(p_train,open('data/cifar100/train_combined.pkl','wb'))
        p.dump(p_test,open('data/cifar100/test_combined.pkl','wb'))


def ssl(dataloader, f_name, SEI = 4):
    l = []
    if SEI == 4:
        for img, cls_tar in dataloader:
            l.append([img,cls_tar,0])
            l.append([np.rot90(img, k=1),cls_tar,1])
            l.append([np.rot90(img, k=2),cls_tar,2])
            l.append([np.rot90(img, k=3),cls_tar,3])

        with open(f_name , 'wb') as f:
            p.dump(l, f)

    elif SEI == 8 :
        for img, cls_tar in dataloader:
            l.append([img,cls_tar,0])
            l.append([np.fliplr(img),cls_tar,0])
            l.append([np.rot90(img, k=1),cls_tar,1])
            l.append([np.rot90(np.fliplr(img), k=1),cls_tar,1])
            l.append([np.rot90(img, k=2),cls_tar,2])
            l.append([np.rot90(np.fliplr(img), k=2),cls_tar,2])
            l.append([np.rot90(img, k=3),cls_tar,3])
            l.append([np.rot90(np.fliplr(img), k=3),cls_tar,3])
        with open(f_name , 'wb') as f:
            p.dump(l, f)


if __name__ == "__main__":
    
    dsets = ['cifar10','cifar100']
    
    for dset in dsets:
        data_download_preprocess(dset)
        path = f'data/{dset}'
        train = p.load(open(f'data/{dset}/train_combined.pkl','rb'))
        test = p.load(open(f'data/{dset}/test_combined.pkl','rb'))
        ssl(train, os.path.join(path,'ssl_train_combined.pkl'),SEI=4)
        ssl(test, os.path.join(path,'ssl_test_combined_SEI4.pkl'),SEI=4)
        ssl(test, os.path.join(path,'ssl_test_combined_SEI8.pkl'),SEI=8)
        print(f'4-way augmented dataset for {dset} completed')
    
