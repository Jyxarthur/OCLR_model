import os
import torch
import einops
import cv2
import json
import numpy as np
import torch.nn.functional as F

from PIL import Image
from itertools import permutations
from scipy.optimize import linear_sum_assignment


""" Data summary and saving """

class AverageMeter(object):
    # Computes averages in terms of categories/frames
    def __init__(self):
        self.variable_dic = {}
        self.variable_dicfloat = {}

    def update(self, variable, category, filename, value):
        if variable not in self.variable_dic.keys():
            self.variable_dic[variable] = {}
        if category not in self.variable_dic[variable].keys():
            self.variable_dic[variable][category] = {}
        if filename not in self.variable_dic[variable][category].keys():
            self.variable_dic[variable][category][filename] = value
        else:
            self.variable_dic[variable][category][filename] = (self.variable_dic[variable][category][filename] + value) / 2
    
    def summary(self, dataset):
        if dataset in ['Syn']:
            meaniou_m = self.average('iou_m', 'framewise')
            self.detail('iou_m')
            meaniou_am = self.average('iou_am', 'framewise')
            self.detail('iou_am')
        elif dataset in ['DAVIS16', 'DAVIS17m', 'Segtrack']:
            meaniou_m = self.average('iou_m', 'category', dataset)
            meaniou_am = -1 
            self.detail('iou_m')
        elif dataset in ['FBMS', 'MoCA']:
            print("Please save the visualisation results and evaluate use official evaluators")
            meaniou_m = -1  
            meaniou_am = -1 
        else: 
            meaniou_m = self.average('iou_m', 'framewise')
            meaniou_am = self.average('iou_am', 'framewise')
            self.detail('iou_m')
            self.detail('iou_am')
        return meaniou_m, meaniou_am
        
    def detail(self, variable):
        variable_info = self.variable_dic[variable]
        print("---- Summary of " + variable)
        for category in variable_info.keys():
            info_mean = variable_info[category].values()
            print('      ' + category + ': ' + str(np.around(sum(info_mean)/ len(info_mean), decimals=5)))                
    
    def average(self, variable, mode, note = 'DAVIS17m'):
        if variable not in self.variable_dic.keys():
            print("Variable not available")
        variable_info = self.variable_dic[variable]
        full_info_list = []
        
        if mode == 'category' and note != 'DAVIS17m':
            for category in variable_info.keys():
                info_list = list(variable_info[category].values())
                info_mean = sum(info_list) / len(info_list)
                full_info_list.append(info_mean)
            if len(full_info_list) == 0:
                print("Empty variable recording of --" + variable)
                return None
            variable_info_mean = sum(full_info_list) / len(full_info_list)
            
        elif mode == 'framewise':
            for category in variable_info.keys():
                info_list = list(variable_info[category].values())
                full_info_list.extend(info_list)
            if len(full_info_list) == 0:
                print("Empty variable recording of --" + variable)
                return None
            variable_info_mean = sum(full_info_list) / len(full_info_list)
            
        elif mode == 'category' and note == 'DAVIS17m': 
            # For IoU evaluation, we take the average over frames. However, DAVIS17-motion dataset evaluation takes an average over objects.
            # Therefore, below calculates the object-averaged IoU by putting more weights on multiple object dataset
            for category in variable_info.keys():
                info_list = list(variable_info[category].values())
                info_mean = sum(info_list) / len(info_list)
                if category in ['bike-packing', 'judo']:
                    for i in range(2):
                        full_info_list.append(info_mean)
                elif category in ['dogs-jump', 'india', 'loading', 'pigs']:
                    for i in range(3):
                        full_info_list.append(info_mean)
                elif category in ['gold-fish']:
                    for i in range(5):
                        # Since OCLR predicts only 3 channels (and IoU averaged over 3 channels), while 5 objects are present in "gold-fish" dataset
                        # A "0.6" factor is used to correct the problem
                        full_info_list.append(0.6 * info_mean)
                else:
                    full_info_list.append(info_mean)
            if len(full_info_list) == 0:
                print("Empty variable recording of --" + variable)
                return None
            variable_info_mean = sum(full_info_list) / len(full_info_list)
        variable_info_mean = np.around(variable_info_mean, decimals = 5)
        print(" ")
        print(variable + ' ' + mode + ' ' + str(variable_info_mean))
        return variable_info_mean


def imwrite_indexed(filename, array, colour_palette):
    # Save indexed png for DAVIS
    im = Image.fromarray(array)
    im.putpalette(colour_palette.ravel())
    im.save(filename, format='PNG')
    
def save_indexed(filename, img, colours = [[128, 0, 0], [0, 128, 0], [128, 128, 0]]):
    colour_palette = np.array([[0,0,0]] + colours).astype(np.uint8)
    imwrite_indexed(filename, img, colour_palette)


def save_vis_results(mask_m, mask_am, gt_res, dataset, category, filenames, save_path):
    with open('data/colour_palette.json') as f:
        colour_dict = json.load(f)
    if dataset in colour_dict.keys():    
        colours = colour_dict[dataset]
    else:
        colours = [[128, 0, 0], [0, 128, 0], [128, 128, 0]]
    if dataset in ['DAVIS16', 'FBMS', 'Segtrack', 'MoCA']:
        mask_m = (mask_m[:, :, 0:1] > 0.5).float() 
        mask_am = None
    elif dataset in ['Syn']:
        mask_m = (mask_m > 0.5).float() 
        mask_am = mask_am
    else: # dataset in ['DAVIS17m']:
        mask_m = (mask_m > 0.5).float() 
        mask_am = None
        
    b, t, c, h, w = mask_m.size()
    H, W = gt_res
    for i in range(mask_m.size()[0]):
        for k in range(t):
            multimask = torch.clone(mask_m[0, 0, 0]).cpu().detach().numpy()
            multimask = np.repeat(multimask[:, :, np.newaxis], 3, 2)
            multimask = cv2.resize(multimask, (W, H))
            multimask[:, :, :] = 0
            for j in range(c):
                singlemask = mask_m[i, k, j].cpu().detach().numpy() 
                singlemask = np.repeat(singlemask[:, :, np.newaxis], 3, 2)
                offset = np.resize(np.array(colours[j]), (H, W, 3))
                singlemask = cv2.resize(singlemask, (W, H))
                singlemask = (singlemask > 0.5).astype(np.float32)
                singlemask *= (j + 1)
                multimask += singlemask
            multimask = np.clip(multimask[:, :, 0], 0, 3)
            multimask = multimask.astype(np.uint8)
            os.makedirs(os.path.join(save_path, dataset + '_m', category), exist_ok=True)
            save_indexed(os.path.join(save_path, dataset + '_m', category, filenames[k]), multimask, colours)
            
            if mask_am is not None:
                for j in range(c):
                    singleammask = mask_am[i, k, j].cpu().detach().numpy() 
                    singleammask = np.repeat(singleammask[:, :, np.newaxis], 3, 2)
                    singleammask = (singleammask * 255).astype(np.uint8)
                    os.makedirs(os.path.join(save_path, dataset + '_am', category), exist_ok=True)
                    cv2.imwrite(os.path.join(save_path, dataset + '_am', category, str(j+1) + "_" + filenames[k]), singleammask)




""" Modal construction and ordering related """

def find_recon_mask(mask, order):
    # Get modal masks from amodal masks and an order
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    b, t, c, h, w = mask.size()
    # c = order.size()
    recon_mask = torch.clone(mask)
    acc_mask = torch.zeros(b, t, h, w).to(device)
    for channel in reversed(order):
        recon_mask[:, :, channel] = mask[:, :, channel] * (1 - acc_mask) 
        acc_mask = (1 - acc_mask) * mask[:, :, channel] + acc_mask
    return recon_mask


def find_recon_factor(order_raw):
    # Obtain possibility of each order permutation based on predicted ordering
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    b, t, c = order_raw.size()
    order_raw_rel = order_raw.unsqueeze(3) - order_raw.unsqueeze(2) # b t c c
    tuidx = torch.triu_indices(c, c, 1).to(device)
    tmpr = 1
    order_raw_rel = order_raw_rel[:, :, tuidx[0], tuidx[1]] / tmpr # b t c
    order_raw = torch.clamp(torch.sigmoid(order_raw_rel), 1e-5, 1-1e-5)
    recon_fac = torch.prod(order_raw, 2)
    return recon_fac
    
def amodal_to_modal_soft(mask_am, order_raw):
    # Construction of modal masks from amodal masks and orders with soft thersholding (used for training)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    perm_orders = list(permutations(range(mask_am.size()[2])))  # obtain all possible permutation order
    perm_orders = torch.tensor(perm_orders).long().to(device)
    # Reconstruct modal masks from amodal masks and **all** possible order permutations
    for i, order in enumerate(perm_orders):
        order_raw_tmp = order_raw[:, :, order]
        recon_fac_tmp =  find_recon_factor(order_raw_tmp) # possibility of each reconstructed mask
        recon_mask_tmp = find_recon_mask(mask_am, order) # reconstructed mask by a particular order permutation
        if i == 0:
            recon_facs = recon_fac_tmp.unsqueeze(0)
            recon_masks = recon_mask_tmp.unsqueeze(0)
        else:
            recon_facs = torch.cat((recon_facs, recon_fac_tmp.unsqueeze(0)), 0)
            recon_masks = torch.cat((recon_masks, recon_mask_tmp.unsqueeze(0)), 0)
    e, b, t, c, h, w = recon_masks.size()
    recon_facs = recon_facs / torch.sum(recon_facs, 0).unsqueeze(0) # normalise reconstruction factor
    recon_mask = torch.sum(recon_facs[:, :, :, None, None, None].expand(e, b, t, c, h, w) * recon_masks, 0)
    recon_mask = torch.clamp(recon_mask, 0., 1.)
    return recon_mask
    
def amodal_to_modal_hard(mask_am, order_raw, thres = 0.5):
    # Construction of modal masks from amodal masks and orders with hard thersholding (used for inference)
    b, t, c = order_raw.size()
    mask_am = (mask_am > thres).float()
    mask_m = torch.clone(mask_am)
    _, order = torch.sort(-order_raw, 2)
    order = order.long()
    for i in range(b):
        for j in range(t):
            mask_m[i:i+1, j:j+1] = find_recon_mask(mask_am[i:i+1, j:j+1], order[i, j])
    return mask_m



""" Evaluation metrics """

def iou(masks, gt, thres=0.5):
    masks = (masks>thres).float()
    gt = (gt>thres).float()
    intersect = (masks * gt).sum(dim=[-2, -1])
    union = masks.sum(dim=[-2, -1]) + gt.sum(dim=[-2, -1]) - intersect
    empty = (union < 1e-6).float()
    iou = torch.clip(intersect/(union + 1e-12) + empty, 0., 1.)
    return iou
    

def hungarian_iou(masks, gt): 
    T, C, H, W = gt.size()
    masks = F.interpolate(masks, size=(H, W)) 
    masks = masks.unsqueeze(1)
    gt = gt.unsqueeze(2)
    mean_ious = []
    IoUs = iou(masks, gt).cpu().detach().numpy()
    framemean_IoU = np.mean(IoUs, 0)
    indices = linear_sum_assignment(-framemean_IoU)
    exist_list = []
    for c in range(C):
        volume = torch.sum(gt[:, c]) 
        if volume / (T * H * W) > 1e-6:
            exist_list.append(c)
    for b in range(masks.size()[0]):
        total_iou = 0
        IoU = iou(masks[b], gt[b]).cpu().detach().numpy()
        for idx in range(indices[0].shape[0]):
            i = indices[0][idx]
            if i not in exist_list:
                continue
            j = indices[1][idx]
            total_iou += IoU[i, j]
        if len(exist_list) == 0:
            mean_iou = 0
        else:
            mean_iou = total_iou / (len(exist_list))
        mean_ious.append(mean_iou)
    return mean_ious


def hungarian_matcher(mask_am_raw, gt_am, varlist):
    # Hungarian matching of variable lists (varlist) based on **cross-entropy loss** between mask_am_raw and gt_am
    # Here, the input mask (mask_am_raw) requires a further sigmoid to constrain the range within [0, 1]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    b, t, c, h, w = mask_am_raw.size()
    maskam_match = torch.clone(mask_am_raw)
    maskam_match = maskam_match.sigmoid()
    gt_am = einops.rearrange(gt_am, 'b t c h w -> (b t) c h w')
    gt_am = F.interpolate(gt_am, size=(h, w))      # to save evalation time, downsize the gt masks to match with the output size
    gt_am = einops.rearrange(gt_am, '(b t) c h w -> b t c h w', b = b)
    gt_am = torch.clamp(gt_am, 0., 1.)
    gtam_match = torch.clone(gt_am)
    maskam_match = maskam_match.unsqueeze(2) # b t 1 c h w 
    gtam_match = gtam_match.unsqueeze(3) # b t c 1 h w
    
    matched_varlist = []
    for var in varlist: 
        matched_var = torch.clone(var).to(device) # b t c ...
        matched_varlist.append(matched_var)
        
    for b_i in range(b):
        amloss = F.binary_cross_entropy(maskam_match[b_i].expand(t, c, c, h, w), gtam_match[b_i].expand(t, c, c, h, w), reduction = "none")
        amloss = torch.nan_to_num(amloss, nan = 10., posinf = 10., neginf = 10.)  # avoid exceptions
        amloss = amloss.mean(dim = [-1, -2]).cpu().detach().numpy()
        framemean_amloss = np.mean(amloss, 0)
        index = linear_sum_assignment(framemean_amloss)
        permute_index = torch.tensor(index[1].tolist()).to(device)
        for matched_var in matched_varlist:
            matched_var[b_i] = matched_var[b_i, :, permute_index]
    return matched_varlist


def hungarian_matcher_iou(mask_am, gt_am, varlist):
    # Hungarian matching of variable lists (varlist) based on **IoUs** between mask_am_raw and gt_am
    # Here, the input mask (mask_am_raw) has the range within [0, 1] (no additional sigmoid required)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    b, t, c, h, w = mask_am.size()
    maskam_match = torch.clone(mask_am)
    gt_am = einops.rearrange(gt_am, 'b t c h w -> (b t) c h w')
    gt_am = F.interpolate(gt_am, size=(h, w))      # to save evalation time, downsize the gt masks to match with the output size
    gt_am = einops.rearrange(gt_am, '(b t) c h w -> b t c h w', b = b)
    gt_am = torch.clamp(gt_am, 0., 1.)
    gtam_match = torch.clone(gt_am)
    maskam_match = maskam_match.unsqueeze(2) # b t 1 c h w 
    gtam_match = gtam_match.unsqueeze(3) # b t c 1 h w
    
    matched_varlist = []
    for var in varlist: 
        matched_var = torch.clone(var).to(device) # b t c ...
        matched_varlist.append(matched_var)
        
    for b_i in range(b):
        IoUs = iou(maskam_match[b_i].expand(t, c, c, h, w), gtam_match[b_i].expand(t, c, c, h, w)).cpu().detach().numpy()
        framemean_IoU = np.mean(IoUs, 0)        
        index = linear_sum_assignment(-framemean_IoU)
        permute_index = torch.tensor(index[1].tolist()).to(device)
        for matched_var in matched_varlist:
            matched_var[b_i] = matched_var[b_i, :, permute_index]
    return matched_varlist
    

""" Training related functions """

def set_learning_rate(optimizer, lr):
    for g in optimizer.param_groups:
        g['lr'] = lr


def criterion_amodal_bce(mask_am_raw, gt_am):
    b, t, c, h, w = mask_am_raw.size()
    gt_am = einops.rearrange(gt_am, 'b t c h w -> (b t) c h w')
    gt_am = F.interpolate(gt_am, size=(h, w))   # resize gt amodal mask size to match the prediction, in order to save training time
    mask_am_raw = einops.rearrange(mask_am_raw, 'b t c h w -> (b t) c h w')
    loss_amodal_bce = F.binary_cross_entropy_with_logits(mask_am_raw, gt_am)
    return loss_amodal_bce


def find_boundary(mask, in_boundary = 3, out_boundary = 3):
    # find boundary region (+- 3 pixels) of the mask
    # mask: (b t) c h w
    n = out_boundary
    mask_out = torch.clone(mask)
    mask_out[:, :, :, n:] = torch.clamp(mask[:, :, :, :-n] + mask_out[:, :, :, n:], 0, 1)
    mask_out[:, :, :, :-n] = torch.clamp(mask[:, :, :, n:] + mask_out[:, :, :, :-n], 0, 1)
    mask_out[:, :, n:, :] = torch.clamp(mask_out[:, :, :-n, :] + mask_out[:, :, n:, :], 0, 1)
    mask_out[:, :, :-n, :] = torch.clamp(mask_out[:, :, n:, :] + mask_out[:, :, :-n, :], 0, 1)
    m = in_boundary
    mask_in = torch.clone(mask)
    mask_in[:, :, :, m:] = (((mask[:, :, :, :-m] + mask_in[:, :, :, m:])/2) == 1.).float()
    mask_in[:, :, :, :-m] = (((mask[:, :, :, m:] + mask_in[:, :, :, :-m])/2) == 1.).float()
    mask_in[:, :, m:, :] = (((mask_in[:, :, :-m, :] + mask_in[:, :, m:, :])/2) == 1.).float()
    mask_in[:, :, :-m, :] = (((mask_in[:, :, m:, :] + mask_in[:, :, :-m, :])/2) == 1.).float()
    mask_boundary =  mask_out - mask_in
    mask_boundary = (mask_boundary > 0.5).float()
    return mask_boundary
    
    
def criterion_amodal_bound(mask_am_raw, gt_am):
    b, t, c, h, w = mask_am_raw.size()
    gt_am = einops.rearrange(gt_am, 'b t c h w -> (b t) c h w')
    gt_am = F.interpolate(gt_am, size=(h, w))
    mask_boundary = find_boundary(gt_am)
    gt_am_boundary = gt_am * mask_boundary
    
    mask_am_raw = einops.rearrange(mask_am_raw, 'b t c h w -> (b t) c h w')
    mask_am_boundary = mask_am_raw.sigmoid() * mask_boundary
    
    loss_bound = F.binary_cross_entropy(mask_am_boundary, gt_am_boundary)
    loss_bound = loss_bound / (torch.sum(mask_boundary) + 1e-12) * (b * t * c * h * w)
    return loss_bound


def criterion_modal(mask_m, gt_m):
    b, t, c, h, w = mask_m.size()
    gt_m = einops.rearrange(gt_m, 'b t c h w -> (b t) c h w')
    gt_m = F.interpolate(gt_m, size=(h, w))
    mask_m = einops.rearrange(mask_m, 'b t c h w -> (b t) c h w')
    loss_modal = F.binary_cross_entropy(mask_m, gt_m)
    return loss_modal
    


def criterion_order(order_raw, mask_am):
    # This code only works for our synthetic data, where gt order of layers are fixed and as the same as the channel order of gt amodal mask inputs.
    
    # Below only check if there are overlaps between pairs of layers. 
    # An overlap would suggest that the channel on the front occludes the channel on the back. Otherwise, no order relationship can be judged.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    b, t, c, h, w = mask_am.size()
    b, t, c = order_raw.size()
    
    if c == 1: # Order loss only work when there are multiple layers
        return torch.tensor(0.)
    else:
        mask_1 = mask_am.unsqueeze(2)  # b t 1 c h w
        mask_2 = mask_am.unsqueeze(3)  # b t c 1 h w
        inter = (mask_1 * mask_2).sum(dim = [-2, -1]) # b t c c, check if there are overlaps between every pairs of channels
        fil = 1. - torch.eye(c)
        fil = fil.unsqueeze(0).unsqueeze(0).to(device) # 1 1 c c, which filters out diagonal elements
        inter *= fil
        inter = (torch.sum(inter, 1) > 0).float() / 2 + 0.5 # b c c
        tuidx = torch.triu_indices(c, c, 1)
        # For gt order, for example, if b = 1, c = 3, the three entries represent whether there is overlap between layer 1-2, layer 2-3, layer 1-3, respectively.
        # If there is an overlap, the entry = 1 (representing occlusion), otherwise the entry = 0.5 (representing no order relationship)
        gt_order = inter[:, tuidx[0], tuidx[1]] # b c
        order_raw = torch.mean(order_raw, 1) # b c, to obtain the ordering for the whole sequence
        order_rel = order_raw.unsqueeze(2) - order_raw.unsqueeze(1) # b c c
        tuidx = torch.triu_indices(c, c, 1).to(device)
        tmpr = 1
        order_rel = order_rel[:, tuidx[0], tuidx[1]] / tmpr # b c, obtaining the pseudo-depth difference between every pairs of layers
        order_rel = torch.clamp(torch.sigmoid(order_rel), 1e-5, 1-1e-5)
        loss_order = torch.mean(-(1 - (gt_order == 0.5).float()) * (gt_order * torch.log(order_rel) + (1 - gt_order) * torch.log(1 - order_rel)))
        return loss_order



### from: https://github.com/pytorch/pytorch/issues/15849#issuecomment-518126031
class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

# https://github.com/pytorch/pytorch/issues/15849#issuecomment-573921048
class FastDataLoader(torch.utils.data.dataloader.DataLoader):
    '''for reusing cpu workers, to save time'''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        # self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)
