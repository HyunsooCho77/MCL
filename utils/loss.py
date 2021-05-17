import torch
import torch.nn.functional as F
import torch.nn as nn

def nt_xent(x, t=0.5):
    # l2 norm first
    x = F.normalize(x, dim=1)
    
    # calculate cosine-sim between all possible pairs in minibatch
    x_scores =  (x @ x.t()).clamp(min=1e-7)  # normalized cosine similarity scores
    x_scale = x_scores / t   # scale with temperature

    # (2N-1)-way softmax without the score of i-th entry itself.
    # Set the diagonals to be large negative values, which become zeros after softmax.
    x_scale = x_scale - torch.eye(x_scale.size(0)).to(x_scale.device) * 1e5
    # targets 2N elements.
    targets = torch.arange(x.size()[0])
    targets[::2] += 1  # target of 2k element is 2k+1
    targets[1::2] -= 1  # target of 2k+1 element is 2k
    return F.cross_entropy(x_scale, targets.long().to(x_scale.device))


def nt_xent_CCM(x, y, beta = 2, alpha = 0.05, t=0.2):
    # l2 norm first
    x = F.normalize(x, dim=1)
    c = [y]*len(y)
    
    mask = torch.stack(c)
    for i in range(len(mask)):
        mask[i] = mask[i]- mask[i][i]
    
    mask_pos = (mask>0).type(torch.float)  /t
    mask_neg = (mask<0).type(torch.float)  /t
    
    mask_rot_eq = (mask%10==0).type(torch.float) * ( beta - 1/t )
    mask_eq = (mask==0).type(torch.float) * (1/t - beta + alpha)
    
    diag = torch.eye(len(y)) * (1/t - alpha)
    mask_sum = mask_pos + mask_neg + diag + mask_rot_eq + mask_eq

    mask_sum = mask_sum.view(1,1,len(y),len(y))
    mask_final = F.interpolate(mask_sum,scale_factor = 2,mode='nearest')
    mask_final = mask_final.squeeze().cuda()

    # calculate cosine-sim between all possible pairs in minibatch
    x_scores =  (x @ x.t()).clamp(min=1e-7)  # normalized cosine similarity scores


    # apply mask to calculated similarity
    x_scale = x_scores * mask_final

    # (2N-1)-way softmax without the score of i-th entry itself.
    # Set the diagonals to be large negative values, which become zeros after softmax.
    x_scale = x_scale - torch.eye(x_scale.size(0)).to(x_scale.device) * 1e5

    # targets 2N elements.
    targets = torch.arange(x.size()[0])
    targets[::2] += 1  
    targets[1::2] -= 1  
    
    return F.cross_entropy(x_scale, targets.long().to(x_scale.device))


def nt_xent_SPA(x,y,beta=2, alpha=0.05, t=0.2):
    # l2 norm first
    x = F.normalize(x, dim=1)

    #################attrack mask
    batch_size = len(y)
    label_info = [y]*batch_size

    att_mask = torch.stack(label_info)
    for i in range(len(att_mask)):
        att_mask[i] = att_mask[i]- att_mask[i][i]
        att_mask[i][i] = att_mask[i][i] - 1

    same_class_mask = (att_mask == 0).type(torch.uint8)
    
    label = torch.Tensor([]).long()
    for idx, row in enumerate(same_class_mask):
        random_choice = torch.nonzero(row).squeeze()
        random_idx = torch.randint(random_choice.size()[0], (1,)) 
        label = torch.cat((label,random_choice[random_idx])) 


    label = label * 2
    
    label_copy1 = torch.unsqueeze(label, 1)
    label_copy2 = torch.unsqueeze(label, 1)
    concatted = torch.cat([label_copy1, label_copy2], 1)
    result = concatted.view([-1, batch_size*2]).squeeze()
    result[1::2]+=1

    #repel mask
    c = [y]*len(y)
    mask = torch.stack(c)
    for i in range(len(mask)):
        mask[i] = mask[i]- mask[i][i]
    
    mask_pos = (mask>0).type(torch.uint8) /t
    mask_neg = (mask<0).type(torch.uint8) /t
    
    mask_rot_eq = (mask%10==0).type(torch.float) * ( beta - 1/t )
    mask_eq = (mask==0).type(torch.float) * (1/t - beta + alpha)
    
    diag = torch.eye(len(y)) * - alpha

    mask_sum = mask_pos + mask_neg + mask_rot_eq + mask_eq + diag
    
    mask_sum = mask_sum.view(1,1,len(y),len(y)).type(torch.FloatTensor)
    mask_final = F.interpolate(mask_sum,scale_factor = 2,mode='nearest')
    mask_final = mask_final.squeeze().cuda()

    for i in range(len(mask_final)):
        mask_final[i][result[i]] = 1/t

    # calculate cosine-sim between all possible pairs in minibatch
    x_scores =  (x @ x.t()).clamp(min=1e-7)  # normalized cosine similarity scores

    # (2N-1)-way softmax without the score of i-th entry itself.
    # Set the diagonals to be large negative values, which become zeros after softmax.
    x_scale = x_scores * mask_final
    x_scale = x_scale - torch.eye(x_scale.size(0)).to(x_scale.device) * 1e5
    
    return F.cross_entropy(x_scale, result.long().to(x_scale.device))