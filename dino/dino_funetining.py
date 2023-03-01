import os
import torch 
import einops
import numpy as np
from torch.utils.data import Dataset
from torch.nn import functional as F

import utils
import vision_transformer as vits
from data import extract_feature_batch, read_seg, read_frame, FramePairs, FastDataLoader 


def dino_finetuning(args, frame_list, model, ftckpt_path):
    # learning rates
    lr = 1e-5
    # number of iterations
    num_it = 1000
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    video_dataset = FramePairs(args, frame_list)
    video_loader = FastDataLoader(
            video_dataset, num_workers=8, batch_size=1, shuffle=True, pin_memory=True, drop_last=False)
    it = 0
    losses = []
    while it < num_it:
        for idx, val_sample in enumerate(video_loader):
            rgbs, segs = val_sample # b t c H W, b t 1 c h w
            # t=2 representing a pair of frames (frame0 and frame1)
            b, t, _, c, h, w = segs.size()
            segs = segs[:, :, 0].cuda()
            rgbs = einops.rearrange(rgbs, 'b t c H W  -> (b t) c H W').cuda()
            feat_rgbs = extract_feature_batch(model, rgbs)
            feat_rgbs = F.normalize(feat_rgbs, dim=2, p=2)
            feat_rgbs = einops.rearrange(feat_rgbs, '(b t) hw c -> b t hw c', b = b)
            # find affinity map between frame0 and frame1
            aff = (torch.bmm(feat_rgbs[:, 0], feat_rgbs[:, 1].permute(0, 2, 1)) / 0.07).softmax(2)
            aff = aff.unsqueeze(1).repeat(1, 3, 1, 1)  # b 3 hw0 hw1
            # find positive and negative regions for both frames
            mask_posi, mask_nega = find_regions(segs[:, :, 1:]) # b t c h w 
            mask_posi = einops.rearrange(mask_posi, 'b t c h w  -> b t c (h w)')
            mask_nega = einops.rearrange(mask_nega, 'b t c h w  -> b t c (h w)')
            # find contrastive loss
            loss = 0.5 * (find_loss0(aff, mask_posi, mask_nega, h, w) + find_loss1(aff, mask_posi, mask_nega, h, w))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.detach().cpu().numpy())
            if it > num_it:
                break
            if it >= 1:
                loss_len = len(losses)
                # average loss for first 100 iterations
                acc_loss_0 = sum(losses[:min(100, loss_len)]) / min(100, loss_len)
                # average loss for lastest 100 iterations
                acc_loss = sum(losses[-100:]) / len(losses[-100:])
                if it % 100 == 0:
                    print("Iteration " + str(it) + " with loss " + str(np.around(acc_loss, 4)) + " compare to " + str(np.around(acc_loss_0, 4)))
            
            # training schedule settings
            if it < 100:
                lr = 1e-5 * it / 100
                for g in optimizer.param_groups:
                    g['lr'] = lr
            it += 1
    # save the finetuned checkpoint of DINO transformer
    if ftckpt_path is not None:
        print("Saving the DINO model")
        os.makedirs(ftckpt_path, exist_ok=True)
        filename = os.path.join(ftckpt_path, 'dino_deitsmall8_dinoft_' + str(os.path.basename(os.path.dirname(frame_list[0]))) + '.pth')
        torch.save(model.state_dict(), filename)
    return model

def find_regions(mask, inner_bound = 1, outer_bound = 5):
    """
    Define trimap regions as positive, negative and uncertain regions
    Inner_bound and outer_bound correspond to the distance (number of pixels) between defined trimap boundaries and input masks
    """
    m = inner_bound
    mask_inner = torch.clone(mask)
    mask_inner[:, :, :, :, m:] = (((mask[:, :, :, :, :-m] + mask_inner[:, :, :, :, m:])/2) == 1.).float()
    mask_inner[:, :, :, :, :-m] = (((mask[:, :, :, :, m:] + mask_inner[:, :, :, :, :-m])/2) == 1.).float()
    mask_inner[:, :, :, m:, :] = (((mask_inner[:, :, :, :-m, :] + mask_inner[:, :, :, m:, :])/2) == 1.).float()
    mask_inner[:, :, :, :-m, :] = (((mask_inner[:, :, :, m:, :] + mask_inner[:, :, :, :-m, :])/2) == 1.).float()
    n = outer_bound
    mask_outer = torch.clone(mask)
    mask_outer[:, :, :, :, n:] = torch.clamp(mask[:, :, :, :, :-n] + mask_outer[:, :, :, :, n:], 0, 1)
    mask_outer[:, :, :, :, :-n] = torch.clamp(mask[:, :, :, :, n:] + mask_outer[:, :, :, :, :-n], 0, 1)
    mask_outer[:, :, :, n:, :] = torch.clamp(mask_outer[:, :, :, :-n, :] + mask_outer[:, :, :, n:, :], 0, 1)
    mask_outer[:, :, :, :-n, :] = torch.clamp(mask_outer[:, :, :, n:, :] + mask_outer[:, :, :, :-n, :], 0, 1)
    # determine whether the object mask is too small (i.e. 2 pixels wide)
    # if so, the inner boundary of trimap (and the positive region) will not exist 
    b, t, c, h, w = mask.size()
    non_empty = (mask.sum(dim = [-1, -2]) > 0).float().unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, h, w)
    # positive region
    mask_posi =  mask_inner * non_empty + mask * (1 - non_empty)
    mask_posi = torch.clamp(mask_posi, 0., 1.)
    # uncertain region
    mask_uncer = (mask_outer - mask_inner) * non_empty + (mask_outer - mask) * (1 - non_empty)
    mask_uncer = torch.clamp(mask_uncer, 0., 1.)
    # negative region
    mask_nega = 1 - mask_uncer - mask_posi
    mask_nega = torch.clamp(mask_nega, 0., 1.)
    return mask_posi, mask_nega

def find_loss0(aff, mask_posi, mask_nega, h, w):
    """
    Find contrastive loss by associating frame0 positive regions and frame1 posi+nega regions
    """
    r0_posi = mask_posi[:, 0] #b c hw0
    r1m_posi = mask_posi[:, 1].unsqueeze(2).repeat(1, 1, h*w, 1) #b c hw0 hw1 
    r1m_nega = mask_nega[:, 1].unsqueeze(2).repeat(1, 1, h*w, 1)
    posi_sample = torch.sum(aff * r1m_posi, -1) * r0_posi
    nega_sample = torch.sum(aff * r1m_nega, -1) * r0_posi
    loss0 = - r0_posi * torch.log(posi_sample / (posi_sample + nega_sample + 1e-8) + 1e-8) 
    loss0 = torch.clamp(loss0, 0., 0.65) # set loss upper bound
    loss0 = torch.sum(loss0) / (torch.sum(r0_posi) + 1e-8)
    return loss0

def find_loss1(aff, mask_posi, mask_nega, h, w):
    """
    Find contrastive loss by associating frame1 positive regions and frame0 posi+nega regions
    """
    r1_posi = mask_posi[:, 1] #b c hw1
    r0m_posi = mask_posi[:, 0].unsqueeze(3).repeat(1, 1, 1, h*w) #b c hw0 hw1 
    r0m_nega = mask_nega[:, 0].unsqueeze(3).repeat(1, 1, 1, h*w)
    posi_sample = torch.sum(aff * r0m_posi, -2) * r1_posi
    nega_sample = torch.sum(aff * r0m_nega, -2) * r1_posi
    loss1 = - r1_posi * torch.log(posi_sample / (posi_sample + nega_sample + 1e-8) + 1e-8) 
    loss1 = torch.clamp(loss1, 0., 0.65) # set loss upper bound
    loss1 = torch.sum(loss1) / (torch.sum(r1_posi) + 1e-8)
    return loss1
    