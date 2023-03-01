"""
Some parts are taken from https://github.com/Liusifei/UVC
Main code structures are taken from https://github.com/facebookresearch/dino with the following copyright
"""
import cv2
import torch
import os
import copy
import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torch.nn import functional as F


@torch.no_grad()
def extract_dino_features(model, frame_tars):
    feat_tars = []
    for frame_tar in frame_tars:
        feat_tar = extract_feature(model, frame_tar)
        feat_tars.append(feat_tar)
    feat_tars = torch.stack(feat_tars, 0)
    return feat_tars

def extract_feature(model, frame, return_h_w=False):
    """Extract one frame feature everytime."""
    out = model.get_intermediate_layers(frame.unsqueeze(0).cuda(), n=1)[0]
    out = out[:, 1:, :]  # we discard the [CLS] token
    h, w = int(frame.shape[1] / model.patch_embed.patch_size), int(frame.shape[2] / model.patch_embed.patch_size)
    dim = out.shape[-1]
    out = out[0].reshape(h, w, dim)
    out = out.reshape(-1, dim)
    if return_h_w:
        return out, h, w
    return out

def extract_feature_batch(model, frame, return_h_w=False):
    """Extract one frame feature for batches."""
    out = model.get_intermediate_layers(frame.cuda(), n=1)[0]
    out = out[:, 1:, :]  # we discard the [CLS] token
    h, w = int(frame.shape[2] / model.patch_embed.patch_size), int(frame.shape[3] / model.patch_embed.patch_size)
    if return_h_w:
        return out, h, w
    return out
 

def read_frame(frame_dir, scale_size=[480]):
    """
    read a single frame & preprocess
    """
    img = cv2.imread(frame_dir)
    ori_h, ori_w, _ = img.shape
    if len(scale_size) == 1:
        if(ori_h > ori_w):
            tw = scale_size[0]
            th = (tw * ori_h) / ori_w
            th = int((th // 64) * 64)
        else:
            th = scale_size[0]
            tw = (th * ori_w) / ori_h
            tw = int((tw // 64) * 64)
    else:
        th, tw = scale_size
    img = cv2.resize(img, (tw, th))
    img = img.astype(np.float32)
    img = img / 255.0
    img = img[:, :, ::-1]
    img = np.transpose(img.copy(), (2, 0, 1))
    img = torch.from_numpy(img).float()
    img = color_normalize(img)
    return img, ori_h, ori_w

def read_seg(seg_dir, factor, scale_size=[480]):
    """
    read segmentation & preprocess
    """
    seg = Image.open(seg_dir)
    _w, _h = seg.size # note PIL.Image.Image's size is (w, h)
    if len(scale_size) == 1:
        if(_w > _h):
            _th = scale_size[0]
            _tw = (_th * _w) / _h
            _tw = int((_tw // 64) * 64)
        else:
            _tw = scale_size[0]
            _th = (_tw * _h) / _w
            _th = int((_th // 64) * 64)
    else:
        _th = scale_size[1]
        _tw = scale_size[0]
    small_seg = np.array(seg.resize((_tw // factor, _th // factor), 0))
    small_seg = torch.from_numpy(small_seg.copy()).contiguous().float().unsqueeze(0)
    return to_one_hot(small_seg), np.asarray(seg)

def color_normalize(x, mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]):
    for t, m, s in zip(x, mean, std):
        t.sub_(m)
        t.div_(s)
    return x

def to_one_hot(y_tensor, n_dims=4):
    """
    Take integer y (tensor or variable) with n dims &
    convert it to 1-hot representation with n+1 dims.
    """
    if(n_dims is None):
        n_dims = int(y_tensor.max()+ 1)
    _,h,w = y_tensor.size()
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(h,w,n_dims)
    return y_one_hot.permute(2, 0, 1).unsqueeze(0)
      

def read_frame_list(video_dir):
    frame_list = [img for img in glob.glob(os.path.join(video_dir,"*.jpg"))]
    frame_list = sorted(frame_list)
    return frame_list

def read_frame_list_fwd(video_dir, n):
    frame_list = [img for img in glob.glob(os.path.join(video_dir,"*.jpg"))]
    frame_list = sorted(frame_list)
    frame_list = frame_list[:n+1]
    frame_list.reverse()
    return frame_list

def read_frame_list_bwd(video_dir, n):
    frame_list = [img for img in glob.glob(os.path.join(video_dir,"*.jpg"))]
    frame_list = sorted(frame_list)
    frame_list = frame_list[n:]
    return frame_list

class FramePairs(Dataset):
    """
    Load pairs of frames within the same sequence
    """
    def __init__(self, args, frame_list):
        frame_pairs = []
        # the max distance between frame pairs is 5 frames
        frame_pairs.extend(list(zip(frame_list[:-1], frame_list[1:])))
        frame_pairs.extend(list(zip(frame_list[:-2], frame_list[2:])))
        frame_pairs.extend(list(zip(frame_list[:-3], frame_list[3:])))
        frame_pairs.extend(list(zip(frame_list[:-4], frame_list[4:])))
        frame_pairs.extend(list(zip(frame_list[:-5], frame_list[5:])))
        self.frame_pairs = frame_pairs
        self.seg_dir = args.seg_path
        self.patch_size = args.patch_size

    def __len__(self):
        return len(self.frame_pairs)

    def __getitem__(self, idx):
        rgbs = []
        segs = []
        for frame_name in self.frame_pairs[idx]:
            rgb, _, _ = read_frame(frame_name)
            rgbs.append(rgb)
            seg_path = os.path.join(self.seg_dir, frame_name.split('/')[-2], frame_name.split('/')[-1].replace("jpg", "png"))
            seg, _= read_seg(seg_path, self.patch_size)
            segs.append(seg)
        rgbs = torch.stack(rgbs, 0)
        segs = torch.stack(segs, 0)
        return rgbs, segs
    
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
            
