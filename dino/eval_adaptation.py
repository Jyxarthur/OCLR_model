"""
Some parts are taken from https://github.com/Liusifei/UVC
Main code structures are taken from https://github.com/facebookresearch/dino with the following copyright
"""
# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import copy
import queue
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from urllib.request import urlopen
from torch.nn import functional as F

import utils
import vision_transformer as vits
from dino_finetuning import dino_finetuning
from data import extract_feature, read_seg, read_frame, read_frame_list, read_frame_list_bwd, read_frame_list_fwd

def norm_mask(mask):
    c, h, w = mask.size()
    for cnt in range(c):
        mask_cnt = mask[cnt,:,:]
        if(mask_cnt.max() > 0):
            mask_cnt = (mask_cnt - mask_cnt.min())
            mask_cnt = mask_cnt/mask_cnt.max()
            mask[cnt,:,:] = mask_cnt
    return mask

 
@torch.no_grad()
def check_consistency(args, model, frame_list, video_dir, first_seg, seg_ori, color_palette):
    """
    Check the framewise consistency by mask propagation
    Note: this is different from semi-supervised VOS, as no GT is used, and the propagtion results
    are not re-used but instead use flow-pred. masks for each round of propagation
    """
    # the queue stores the n preceeding frames
    que = queue.Queue(args.n_last_frames)
    # initialise an error list to record l1 loss for each frame
    error_list = []
    # first frame
    frame1, ori_h, ori_w = read_frame(frame_list[0])
    # extract first frame feature
    frame1_feat = extract_feature(model, frame1).T #  dim x h*w

    mask_neighborhood = None
    for cnt in tqdm(range(1, len(frame_list))):
        frame_tar = read_frame(frame_list[cnt])[0]

        # we use the first segmentation and the n previous ones
        used_frame_feats = [frame1_feat] + [pair[0] for pair in list(que.queue)]
        used_segs = [first_seg] + [pair[1] for pair in list(que.queue)]

        # predict the next frame t by mask propagation
        frame_tar_avg, feat_tar, mask_neighborhood = label_propagation(args, model, frame_tar, used_frame_feats, used_segs, mask_neighborhood)
        # read the flow-pred. frame t
        seg_path = os.path.join(args.seg_path, frame_list[cnt].split('/')[-2], frame_list[cnt].split('/')[-1].replace(".jpg", ".png"))
        framet_seg, _= read_seg(seg_path, args.patch_size)
        framet_seg = framet_seg.cuda()
        
        # evaluate l1 loss between prop. results and flow pred. masks
        _, C, h, w = frame_tar_avg.size()
        error = F.l1_loss(framet_seg[:, 1:], frame_tar_avg[:, 1:]) * h * w / ((torch.sum(frame_tar_avg[:, 1:]) + torch.sum(framet_seg[:, 1:])) / 2 + 1e-6)
        error_list.append(error.detach().cpu().numpy())
        
        # pop out oldest frame if neccessary, for effciency purpose, only 3 frame are used
        if que.qsize() == 3:
            que.get()
            
        # push flow pred. masks into queue (not propagated results)
        seg = copy.deepcopy(framet_seg)
        que.put([feat_tar, seg])
        
    error_list = np.stack(error_list, 0)
    errors = (error_list < np.mean(error_list)).astype(np.float32)
    
    # find most time consistency frame as starting frame for later mask propogation
    n_i, n_f = find_startend_value(errors)
    prop_start_frame = int(1 + (n_i + n_f) / 2)
    return prop_start_frame, error_list

@torch.no_grad()
def eval_video_tracking_with_filter(args, model, frame_list, video_dir, first_seg, seg_ori, color_palette, mean_error):
    """
    Evaluate tracking on a video given flow-pred. masks on first frame and filtered frames
    """
    video_folder = os.path.join(args.output_path, video_dir.split('/')[-1])
    os.makedirs(video_folder, exist_ok=True)

    # the queue stores the n preceeding frames
    que = queue.Queue(args.n_last_frames)
    # first frame
    frame1, ori_h, ori_w = read_frame(frame_list[0])
    # extract first frame feature
    frame1_feat = extract_feature(model, frame1).T #  dim x h*w
    # saving first segmentation
    out_path = os.path.join(video_folder, frame_list[0].split('/')[-1].replace('.jpg', '.png'))
    imwrite_indexed(out_path, seg_ori, color_palette)
    mask_neighborhood = None
    
    for cnt in tqdm(range(1, len(frame_list))):
        frame_tar = read_frame(frame_list[cnt])[0]

        # we use the first segmentation and the n previous ones
        used_frame_feats = [frame1_feat] + [pair[0] for pair in list(que.queue)]
        used_segs = [first_seg] + [pair[1] for pair in list(que.queue)]
 
        # predict the next frame t by mask propagation
        frame_tar_avg, feat_tar, mask_neighborhood = label_propagation(args, model, frame_tar, used_frame_feats, used_segs, mask_neighborhood)
        
        # read the flow-pred. frame t
        seg_path = os.path.join(args.seg_path, frame_list[cnt].split('/')[-2], frame_list[cnt].split('/')[-1].replace(".jpg", ".png"))
        framet_seg, _= read_seg(seg_path, args.patch_size)
        framet_seg = framet_seg.cuda()
        _, C, h, w = frame_tar_avg.size()
        
        # evaluate l1 loss between prop. results and flow pred. masks
        error = F.l1_loss(framet_seg[:, 1:], frame_tar_avg[:, 1:]) * h * w / ((torch.sum(frame_tar_avg[:, 1:]) + torch.sum(framet_seg[:, 1:])) / 2 + 1e-6)
        
        # pop out oldest frame if neccessary
        if que.qsize() == args.n_last_frames:
            que.get()
            
        # dynamic refinement process: push current results or flow-pred. mask into queue
        # determined by the l1 error (evaluating consistency)
        if error < mean_error: # flow-pred. mask being consistent with propagated results
            seg = copy.deepcopy((framet_seg + frame_tar_avg) / 2)
        else:
            seg = copy.deepcopy(frame_tar_avg)
        que.put([feat_tar, seg])

        # upsampling & argmax
        frame_tar_avg = F.interpolate(frame_tar_avg, scale_factor=args.patch_size, mode='bilinear', align_corners=False, recompute_scale_factor=False)[0]
        frame_tar_avg = norm_mask(frame_tar_avg)
        _, frame_tar_seg = torch.max(frame_tar_avg, dim=0)
        
        # saving to disk
        frame_tar_seg = np.array(frame_tar_seg.squeeze().cpu(), dtype=np.uint8)
        frame_tar_seg = np.array(Image.fromarray(frame_tar_seg).resize((ori_w, ori_h), 0))
        frame_nm = frame_list[cnt].split('/')[-1].replace(".jpg", ".png")
        imwrite_indexed(os.path.join(video_folder, frame_nm), frame_tar_seg, color_palette)

def restrict_neighborhood(h, w):
    # we restrict the set of source nodes considered to a spatial neighborhood of the query node (i.e. ``local attention'')
    mask = torch.zeros(h, w, h, w)
    for i in range(h):
        for j in range(w):
            for p in range(2 * args.size_mask_neighborhood + 1):
                for q in range(2 * args.size_mask_neighborhood + 1):
                    if i - args.size_mask_neighborhood + p < 0 or i - args.size_mask_neighborhood + p >= h:
                        continue
                    if j - args.size_mask_neighborhood + q < 0 or j - args.size_mask_neighborhood + q >= w:
                        continue
                    mask[i, j, i - args.size_mask_neighborhood + p, j - args.size_mask_neighborhood + q] = 1

    mask = mask.reshape(h * w, h * w)
    return mask.cuda(non_blocking=True)

def label_propagation(args, model, frame_tar, list_frame_feats, list_segs, mask_neighborhood=None):
    """
    propagate segs of frames in list_frames to frame_tar
    """
    # we only need to extract feature of the target frame
    feat_tar, h, w = extract_feature(model, frame_tar, return_h_w=True)
    return_feat_tar = feat_tar.T # dim x h*w
    ncontext = len(list_frame_feats)
    feat_sources = torch.stack(list_frame_feats) # nmb_context x dim x h*w
    feat_tar = F.normalize(feat_tar, dim=1, p=2)
    feat_sources = F.normalize(feat_sources, dim=1, p=2)
    feat_tar = feat_tar.unsqueeze(0).repeat(ncontext, 1, 1)
    aff = torch.exp(torch.bmm(feat_tar, feat_sources) / 0.1) # nmb_context x h*w (tar: query) x h*w (source: keys)
    if args.size_mask_neighborhood > 0:
        if mask_neighborhood is None:
            mask_neighborhood = restrict_neighborhood(h, w)
            mask_neighborhood = mask_neighborhood.unsqueeze(0).repeat(ncontext, 1, 1)
        aff *= mask_neighborhood
    aff = aff.transpose(2, 1).reshape(-1, h * w) # nmb_context*h*w (source: keys) x h*w (tar: queries)
    tk_val, _ = torch.topk(aff, dim=0, k=args.topk)
    tk_val_min, _ = torch.min(tk_val, dim=0)
    aff[aff < tk_val_min] = 0
    aff = aff / torch.sum(aff, keepdim=True, axis=0)
    list_segs = [s.cuda() for s in list_segs]
    segs = torch.cat(list_segs)
    nmb_context, C, h, w = segs.shape
    segs = segs.reshape(nmb_context, C, -1).transpose(2, 1).reshape(-1, C).T # C x nmb_context*h*w
    seg_tar = torch.mm(segs, aff)
    seg_tar = seg_tar.reshape(1, C, h, w)
    return seg_tar, return_feat_tar, mask_neighborhood

def find_startend_value(errors):
    """ 
    Find most temporal consistent frames by finding the longest temporal window with low errors 
    """
    if np.sum(errors) == 0:
        print("Change thershold")
    else:
        errors = np.array([0] + errors.tolist() + [0])
        differrors = np.diff(errors)
        idxs = np.where(differrors != 0)[0]
        diffidxs = np.diff(idxs)
        ranges = diffidxs[int(errors[0] != 0)::2]
        range_idx= np.argmax(ranges)
        start_idx = idxs[int(int(errors[0] != 0) + range_idx * 2)]
        end_idx = idxs[int(int(errors[0] != 0) + range_idx * 2) + 1] - 1
        return int(start_idx), int(end_idx)

def imwrite_indexed(filename, array, color_palette):
    """ Save indexed png."""
    if np.atleast_3d(array).shape[2] != 1:
      raise Exception("Saving indexed PNGs requires 2D array.")

    im = Image.fromarray(array)
    im.putpalette(color_palette.ravel())
    im.save(filename, format='PNG')


      
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with video object segmentation on DAVIS 2017')
    parser.add_argument('--pretrained_weights', default='dino_deitsmall8_pretrain.pth', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base'], help='Architecture (support only ViT atm).')
    parser.add_argument('--patch_size', default=8, type=int, help='Patch resolution of the model.')
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument("--n_last_frames", type=int, default=7, help="number of preceeding frames")
    parser.add_argument("--size_mask_neighborhood", default=12, type=int,
        help="We restrict the set of source nodes considered to a spatial neighborhood of the query node")
    parser.add_argument("--topk", type=int, default=5, help="accumulate label from top k neighbors")
    parser.add_argument("--bs", type=int, default=6, help="Batch size, try to reduce if OOM")
    
    parser.add_argument('--dataset', default='DAVIS17m', type=str)
    parser.add_argument('--data_path', default='/path/to/dataset/', type=str, help='Path where to read dataset info')
    parser.add_argument('--seg_path', default='/path/to/segmask/', type=str, help='Path where to read flow-pred. masks')
    parser.add_argument('--output_path', default='/path/to/output/', type=str, help='Path where to save segmentations')
    parser.add_argument('--finetune_dino', action='store_true')
    parser.add_argument('--ftckpt_path', default=None, type=str, help='Path where to save finetuned checkpoints')
    
    args = parser.parse_args()

    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))

    # building network
    model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    model.cuda()
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)  
    
    if args.finetune_dino:
        # finetune last two layer of dino transformer
        for names, param in model.named_parameters():
            if 'blocks.10' in names or 'blocks.11' in names:
                param.requires_grad = True
            else:
                param.requires_grad = False
    else:
        args.ftckpt_path = None
        for param in model.parameters():
            param.requires_grad = False

    # dataset information
    dataset = args.dataset
    if dataset == 'DAVIS17m':
        # specify whether it is a single-object or multi-object dataset
        dataset_objs = 'multi'
        video_list = ['bike-packing', 'blackswan', 'bmx-trees', 'breakdance', 'camel', 'car-roundabout', 'car-shadow',
                  'cows', 'dance-twirl', 'dog', 'dogs-jump', 'drift-chicane', 'drift-straight', 'goat', 'gold-fish',
                     'horsejump-high', 'india', 'judo', 'kite-surf', 'lab-coat', 'libby', 'loading', 'mbike-trick',
                    'motocross-jump', 'paragliding-launch', 'parkour', 'pigs', 'scooter-black', 'shooting', 'soapbox']
    
    elif dataset == 'DAVIS16':
        dataset_objs = 'single' 
        video_list = ['dog', 'cows', 'goat', 'camel', 'libby', 'parkour', 'soapbox', 'blackswan', 'bmx-trees', 
                    'kite-surf', 'car-shadow', 'breakdance', 'dance-twirl', 'scooter-black', 'drift-chicane', 
                    'motocross-jump', 'horsejump-high', 'drift-straight', 'car-roundabout', 'paragliding-launch']                 
    
    elif dataset == 'Segtrack':
        dataset_objs = 'single' 
        video_list = ['drift', 'birdfall', 'girl', 'cheetah', 'worm', 'parachute', 'monkeydog',
                    'hummingbird', 'soldier', 'bmx', 'frog', 'penguin', 'monkey', 'bird_of_paradise']   
    
    elif dataset == 'FBMS':
        dataset_objs = 'single' 
        video_list = ['camel01', 'cars1', 'cars10', 'cars4', 'cars5', 'cats01', 'cats03', 'cats06', 
                    'dogs01', 'dogs02', 'farm01', 'giraffes01', 'goats01', 'horses02', 'horses04', 
                    'horses05', 'lion01', 'marple12', 'marple2', 'marple4', 'marple6', 'marple7', 'marple9', 
                    'people03', 'people1', 'people2', 'rabbits02', 'rabbits03', 'rabbits04', 'tennis']

    elif dataset == 'MoCA':
        dataset_objs = 'single' 
        video_list = ['arabian_horn_viper', 'arctic_fox_1', 'arctic_wolf_1', 'black_cat_1', 'crab', 'crab_1', 
                    'cuttlefish_0', 'cuttlefish_1', 'cuttlefish_4', 'cuttlefish_5', 
                    'devil_scorpionfish', 'devil_scorpionfish_1', 'flatfish_2', 'flatfish_4', 'flounder', 
                    'flounder_3', 'flounder_4', 'flounder_5', 'flounder_6', 'flounder_7', 
                    'flounder_8', 'flounder_9', 'goat_1', 'hedgehog_1', 'hedgehog_2', 'hedgehog_3', 
                    'hermit_crab', 'jerboa', 'jerboa_1', 'lion_cub_0', 'lioness', 'marine_iguana', 
                    'markhor', 'meerkat', 'mountain_goat', 'peacock_flounder_0', 
                    'peacock_flounder_1', 'peacock_flounder_2', 'polar_bear_0', 'polar_bear_2', 
                    'scorpionfish_4', 'scorpionfish_5', 'seal_1', 'shrimp', 
                    'snow_leopard_0', 'snow_leopard_1', 'snow_leopard_2', 'snow_leopard_3', 'snow_leopard_6', 
                    'snow_leopard_7', 'snow_leopard_8', 'spider_tailed_horned_viper_0', 
                    'spider_tailed_horned_viper_2', 'spider_tailed_horned_viper_3',
                    'arctic_fox', 'arctic_wolf_0', 'devil_scorpionfish_2', 'elephant', 
                    'goat_0', 'hedgehog_0', 
                    'lichen_katydid', 'lion_cub_3', 'octopus', 'octopus_1', 
                    'pygmy_seahorse_2', 'rodent_x', 'scorpionfish_0', 'scorpionfish_1', 
                    'scorpionfish_2', 'scorpionfish_3', 'seal_2',
                    'bear', 'black_cat_0', 'dead_leaf_butterfly_1', 'desert_fox', 'egyptian_nightjar', 
                    'pygmy_seahorse_4', 'seal_3', 'snowy_owl_0',
                    'flatfish_0', 'flatfish_1', 'fossa', 'groundhog', 'ibex', 'lion_cub_1', 'nile_monitor_1',
                    'polar_bear_1', 'spider_tailed_horned_viper_1']


    color_palette = []
    if dataset_objs == 'single':
        color_palette.append([0, 0, 0])
        color_palette.append([255, 255, 255])
    elif dataset_objs == 'multi':
        for line in urlopen("https://raw.githubusercontent.com/Liusifei/UVC/master/libs/data/palette.txt"):
            color_palette.append([int(i) for i in line.decode("utf-8").split('\n')[0].split(" ")])
    color_palette = np.asarray(color_palette, dtype=np.uint8).reshape(-1,3)
    
    for i, video_name in enumerate(video_list):
        video_name = video_name.strip()
        print(f'[{i}/{len(video_list)}] Begin to segmentate video {video_name}.')
        if 'DAVIS' in dataset:
            video_dir = os.path.join(args.data_path, "JPEGImages/480p", video_name)
        else:
            video_dir = os.path.join(args.data_path, "JPEGImages", video_name)

        frame_list = read_frame_list(video_dir)
        first_seg_path = os.path.join(args.seg_path, frame_list[0].split('/')[-2], frame_list[0].split('/')[-1].replace(".jpg", ".png"))
        first_seg, seg_ori = read_seg(first_seg_path, args.patch_size)
        
        print("Check frame temporal consistency by propagation")
        prop_start_frame, error_list = check_consistency(args, model, frame_list, video_dir, first_seg, seg_ori, color_palette)
        # use as a threshold to filter out flow-pred. masks that are not temporally consistent
        mean_error = np.mean(error_list)

        if args.finetune_dino:
            print("Finetuning DINO transformer")
            model_ft = copy.deepcopy(model)
            model = dino_finetuning(args, frame_list, model_ft, args.ftckpt_path)
        
        print("Forward propagation")
        frame_list_fwd = read_frame_list_fwd(video_dir, prop_start_frame)
        first_seg_path_fwd = os.path.join(args.seg_path, frame_list_fwd[0].split('/')[-2], frame_list_fwd[0].split('/')[-1].replace("jpg", "png"))
        first_seg_fwd, seg_ori = read_seg(first_seg_path_fwd, args.patch_size)
        eval_video_tracking_with_filter(args, model, frame_list_fwd, video_dir, first_seg_fwd, seg_ori, color_palette, mean_error)
        
        print("Backward propagation")
        frame_list_bwd = read_frame_list_bwd(video_dir, prop_start_frame)
        first_seg_path_bwd  = os.path.join(args.seg_path, frame_list_bwd[0].split('/')[-2], frame_list_bwd[0].split('/')[-1].replace("jpg", "png"))
        first_seg_bwd, seg_ori = read_seg(first_seg_path_bwd, args.patch_size)
        eval_video_tracking_with_filter(args, model, frame_list_bwd, video_dir, first_seg_bwd, seg_ori, color_palette, mean_error)