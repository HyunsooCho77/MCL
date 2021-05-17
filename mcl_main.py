import hydra
from omegaconf import DictConfig
import torch
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import CosineAnnealingLR
from models.resnet import resnet18, resnet34, resnet50,resnet101
import torch.nn as nn
from tqdm import tqdm
from models.models import *
from utils.utils import *
from utils.data_loader import *
from train import train_MCL
from evaluate.mcl_validate import *
from sys import exit


@hydra.main(config_path='mcl_config.yml')
def main(args: DictConfig) -> None:

    train_pair_loader = load_trainset(args)

    # Prepare model
    assert args.backbone in ['resnet18', 'resnet34', 'resnet50','resnet101']
    base_encoder = eval(args.backbone)
    model = SimCLR(base_encoder, projection_dim=args.projection_dim)
    
    # Evaluate model
    if args.evaluate == True:
        print('loading pretrained model')
        load_epoch = load_pth(f'ckpt/epoch_{args.load_epoch}.pt', model)
        model = model.cuda()
        train_loader, test_ind_loader, test_ood_loader = load_testset(args)
        aur = SEI(train_loader, test_ind_loader, test_ood_loader, args.load_epoch, model, args)
        exit()

    # optimizer and scheduler (cosine annealing lr)
    learning_rate = 0.3 * (args.batch_size/ 256)
    optimizer = torch.optim.SGD( model.parameters(), lr = learning_rate, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    total_step = args.epochs * len(train_pair_loader)
    scheduler = CosineAnnealingLR(optimizer, T_max= total_step, eta_min=learning_rate * 5e-4) # ratio 3,7 (SGDR)

    assert torch.cuda.is_available()
    cudnn.benchmark = True
    
    #### MCL Training
    model = nn.DataParallel(model).cuda()
    for epoch in range(1, args.epochs+1):
        model.train()
        train_pair_loader = load_trainset(args)
        train_MCL(train_pair_loader,model,optimizer,scheduler,epoch,args)
       
        if epoch % args.log_interval == 0:
            os.makedirs('ckpt',exist_ok = True)
            save_pth(epoch, model, optimizer, scheduler, f'ckpt/epoch_{epoch}.pt')
        
        if epoch % 100 == 0:
            train_loader, test_ind_loader, test_ood_loader = load_testset(args)
            aur = SEI(train_loader, test_ind_loader, test_ood_loader, epoch, model, args)

if __name__ == '__main__':
    main()
   
   

