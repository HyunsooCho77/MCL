from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader, RandomSampler
from PIL import Image
from torchvision import datasets, transforms
import torch
import hydra
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
import pickle
from random import shuffle

# color distortion composed by color jittering and color dropping.
# See Section A of SimCLR: https://arxiv.org/abs/2002.05709

def get_color_distortion(s=0.5):  # 0.5 for CIFAR10 by default
    # s is the strength of color distortion
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort

class ood_dset(Dataset):
    def __init__(self, fname, transform  = None):
        try :
            self.data = pickle.load(open(fname, 'rb'))
        except :
            self.data = fname
        self.transform = transform
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        img, class_label, rot_label = self.data[idx]
        
        img = Image.fromarray(img)
        img = self.transform(img) if self.transform != None else img
        
        return img, class_label, rot_label

class ood_dset_val(Dataset):
    def __init__(self, fname, flip= False, transform  = None):
        try :
            self.data = pickle.load(open(fname, 'rb'))
        except :
            self.data = fname
        self.transform = transform
        self.flip = flip
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        img, class_label, rot_target = self.data[idx]
        
        img = Image.fromarray(img)

        if self.flip == True:
            if rot_target %2 ==0:
                flip = transforms.RandomHorizontalFlip(p=1)
                img = flip(img)
            else :
                flip = transforms.RandomVerticalFlip(p=1)
                img = flip(img)
        
        img = self.transform(img) if self.transform != None else img
        
        return img, class_label, rot_target

class ood_dsetpair(ood_dset):
    """Generate mini-batche pairs on CIFAR10 training set."""

    # contains x4 image (4-way rotations)

    def __getitem__(self, idx):
        img, target, rot_target = self.data[idx]
        img = Image.fromarray(img)  # .convert('RGB')
        if rot_target %2 ==0:
            flip = transforms.RandomHorizontalFlip(p=0.5)
        else :
            flip = transforms.RandomVerticalFlip(p=0.5)

        imgs = [self.transform(flip(img)), self.transform(flip(img))]
        return torch.stack(imgs), target, rot_target  # stack a positive pair


def load_trainset(args) :
    train_transform = transforms.Compose([transforms.RandomResizedCrop(32),
                                          get_color_distortion(s=args.distortion),
                                          transforms.ToTensor()])

    data_dir = hydra.utils.to_absolute_path(args.data_dir)  # get absolute path of data dir

    if args.dataset == 'cifar10':
        print('CIFAR10 dataset!')
        dat = pickle.load(open(os.path.join(data_dir,f'cifar10/ssl_train_combined.pkl'),'rb'))
        
        if args.SEI_shuffle == True:
            print('Shuffling data')
            dat_len = int(len(dat)/4)
            random_list = [*range(dat_len)]
            shuffle(random_list)
            shuffled_dat = []
            for idx in random_list:
                shuffled_dat.append(dat[4*idx])
                shuffled_dat.append(dat[4*idx+1])
                shuffled_dat.append(dat[4*idx+2])
                shuffled_dat.append(dat[4*idx+3])

            train_pair_set = ood_dsetpair(shuffled_dat, transform = train_transform)
            train_pair_loader = DataLoader(train_pair_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=True)
        else :
            train_pair_set = ood_dsetpair(dat, transform = train_transform)
            train_pair_loader = DataLoader(train_pair_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
        
        return train_pair_loader

def load_testset(args) :

    vanilla_test_transform = transforms.Compose([transforms.ToTensor()])
    data_dir = hydra.utils.to_absolute_path(args.data_dir)  # get absolute path of data dir

    if args.dataset == 'cifar10':
        print('CIFAR10 dataset!')
                    
        train_set1 = ood_dset_val(os.path.join(data_dir,f'cifar10/ssl_train_combined.pkl'), flip = False, transform = vanilla_test_transform)
        train_set2 = ood_dset_val(os.path.join(data_dir,f'cifar10/ssl_train_combined.pkl'), flip = True, transform = vanilla_test_transform)
        train_set = torch.utils.data.ConcatDataset([train_set1,train_set2])
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=False)
        
        if args.SEI_way == 4 or args.SEI_way == 1: 
            test_ind_set = ood_dset_val(os.path.join(data_dir,f'cifar10/ssl_test_combined_SEI4.pkl'), flip = False, transform = vanilla_test_transform)
            test_ood_set = ood_dset_val(os.path.join(data_dir,f'cifar100/ssl_test_combined_SEI4.pkl'), flip = False, transform = vanilla_test_transform)
            bsize =4
            
        elif args.SEI_way == 8:
            test_ind_set = ood_dset_val(os.path.join(data_dir,f'cifar10/ssl_test_combined_SEI8.pkl'), flip = True, transform = vanilla_test_transform)
            test_ood_set = ood_dset_val(os.path.join(data_dir,f'cifar100/ssl_test_combined_SEI8.pkl'), flip = False,transform = vanilla_test_transform)
            bsize =8

        test_ind_loader = DataLoader(test_ind_set, batch_size=bsize, shuffle=False, num_workers=args.workers, drop_last=False)
        test_ood_loader = DataLoader(test_ood_set, batch_size=bsize, shuffle=False, num_workers=args.workers, drop_last=False)

    return train_loader, test_ind_loader, test_ood_loader

