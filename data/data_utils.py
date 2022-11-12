import os
import cv2
import json
import einops
import numpy as np



""" Flow Reading """

TAG_FLOAT = 202021.25

def read_flo(file):
    assert type(file) is str, "file is not str %r" % str(file)
    assert os.path.isfile(file) is True, "file does not exist %r" % str(file)
    assert file[-4:] == '.flo', "file ending is not .flo %r" % file[-4:]
    f = open(file,'rb')
    flo_number = np.fromfile(f, np.float32, count=1)[0]
    assert flo_number == TAG_FLOAT, 'Flow number %r incorrect. Invalid .flo file' % flo_number
    w = np.fromfile(f, np.int32, count=1)
    h = np.fromfile(f, np.int32, count=1)
    data = np.fromfile(f, np.float32, count=2*w[0]*h[0])
    flow = np.resize(data, (int(h), int(w), 2))
    f.close()
    return flow

def readFlow(sample_dir, resolution):
    flow = read_flo(sample_dir)
    h, w, _ = np.shape(flow)
    flow = cv2.resize(flow, (resolution[1], resolution[0]), interpolation=cv2.INTER_LINEAR)
    flow[:, :, 0] = flow[:, :, 0] * resolution[1] / w
    flow[:, :, 1] = flow[:, :, 1] * resolution[0] / h
    return einops.rearrange(flow, 'h w c -> c h w')
    

""" RGB/Seg Reading """

def readRGB(sample_dir, resolution):
    rgb = cv2.imread(sample_dir)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb = ((rgb / 255.0) - 0.5) * 2.0
    rgb = cv2.resize(rgb, (resolution[1], resolution[0]), interpolation=cv2.INTER_LINEAR)
    rgb = np.clip(rgb, -1., 1.)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    return einops.rearrange(rgb, 'h w c -> c h w')

def readSeg(sample_dir, resolution = None):
    gt = cv2.imread(sample_dir) 
    if resolution is not None:
        gt = cv2.resize(gt, (resolution[1], resolution[0]), interpolation=cv2.INTER_LINEAR)
    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
    return gt

def readAmodalSeg(sample_dir, resolution = None, out_channels = 3):
    # Extract amodal segmentations
    basename = os.path.basename(sample_dir)
    dirname = os.path.dirname(sample_dir)
    catname = os.path.basename(dirname)
    gts = []
    for i in range(1, out_channels + 1):
        if i <= int(catname[0]):
            gt = readSeg(os.path.join(dirname, str(i) + '_' + basename), resolution)
            gt = gt / 255.
            gt = (gt > 0.5).astype(np.float32)
            gts.append(gt)
        else:
            gts.append(gt * 0.)  
    seg_float = np.stack(gts, 0)
    return seg_float[:, :, :, 0]
    
def processMultiSeg(img, gt_resolution = None, out_channels = 3, dataset = 'Syn'):
    with open('data/colour_palette.json') as f:
        colour_dict = json.load(f)
    if dataset in colour_dict.keys():    
        colours = colour_dict[dataset]
    else:
        colours = [[128, 0, 0], [0, 128, 0], [128, 128, 0]]
    
    masks = []
    colours = colours[0 : min(len(colours), out_channels)]
    
    for colour in colours:
        offset = np.broadcast_to(np.array(colour), (img.shape[0], img.shape[1], 3))
        mask = (np.mean(offset == img, 2) == 1).astype(np.float32)
        mask =  np.repeat(mask[:, :, np.newaxis], 3, 2)
        masks.append(mask)
    for j in range(out_channels):
        masks.append(np.zeros((img.shape[0], img.shape[1], 3)))
        
    masks_raw = masks[0 : out_channels]
    masks_float = []
    for i, mask in enumerate(masks_raw):
        if gt_resolution is not None:
            mask_float = (cv2.resize(mask, (gt_resolution[1], gt_resolution[0]), interpolation=cv2.INTER_LINEAR) > 0.5).astype(np.float32)
        else:
            mask_float = mask
        masks_float.append(mask_float)
    masks_float = np.stack(masks_float, 0)[:, :, :, 0]
    return masks_float



""" Index processing and frame grouping """

def getFrameGroupsList(samples, frame_range, frame_overlap, frame_stride):
    # Input: list of samples, e.g. ["dogs/00000.png", "dogs/00001.png", ...]
    # Output: a list of lists/batches that contains samples, e.g. [["dogs/00000.png", ...], ["dogs/00030.png", ...], ...]
    if len(samples) < frame_range: 
        frame_stride = 1
    frame_span = min(len(samples), frame_range) * frame_stride
    while len(samples) < frame_span and frame_stride > 1:
        frame_stride -= 1
        frame_span = frame_range * frame_stride
    assert frame_stride != 0
    frame_overlap = frame_overlap * frame_stride
    frame_groups = int((len(samples) - frame_overlap) / (frame_span - frame_overlap)) 
    smpl_lists = []
    for i in range(frame_groups):
        for j in range(frame_stride):
            smpl_list = []
            for k in range(frame_range):
                smpl_list.append(samples[min(len(samples) - 1, i * (frame_span - frame_overlap) + j + k * frame_stride)])
            smpl_lists.append(smpl_list)

    if (len(samples) - frame_overlap) % (frame_span - frame_overlap) != 0:
        for j in range(frame_stride):
            smpl_list = []
            for k in range(frame_range):
                smpl_list.append(samples[min(len(samples) - 1, len(samples) - frame_span + j + k * frame_stride)])
            smpl_lists.append(smpl_list)
    return smpl_lists



  