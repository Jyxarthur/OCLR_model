import os
import glob
import numpy as np

from torch.utils.data import Dataset
from .data_utils import getFrameGroupsList, readFlow, readSeg, readAmodalSeg, processMultiSeg


class FlowLoader(Dataset):
    def __init__(self, data_dir, resolution, pair_list, data_seq, dataset, 
                 train = False, out_channels = 3, gt_res = None, frames = 1, amodal = False):
        self.train = train
        self.data_dir = data_dir
        self.dataset = dataset
        self.pair_list = pair_list
        self.out_channels = out_channels
        self.amodal = amodal
        
        self.resolution = resolution   # target resized resolution 
        self.gt_res = gt_res   # original resolution of video frames
        self.frames = frames
        self.stride = 1
        self.frame_overlap = 0
        
        self.samples = []
        
        if data_seq is None:
            data_seq = glob.glob(os.path.join(self.data_dir[1], '*'))
        self.data_seq = data_seq         

        for v in self.data_seq:
            samples = sorted(glob.glob(os.path.join(self.data_dir[1], v, '*.jpg')))
            samples = [os.path.join(x.split('/')[-2], x.split('/')[-1]) for x in samples]
            grouped_samples = getFrameGroupsList(samples, self.frames, self.frame_overlap, self.stride)
            self.samples.extend(grouped_samples)
       
        self.gaps = ['gap{}'.format(i) for i in pair_list]
        self.neg_gaps = ['gap{}'.format(-i) for i in pair_list]
        

        print("Total iterations " + str(len(self.samples)))
        print(" ")

    def readSingleSmpl(self, smpl_name, amodal = True):
        ### Read all information regarding a single video frame
        out = []
        fgap = []
        
        # Read flow information
        for gap, _gap in zip(self.gaps, self.neg_gaps):
            flow_dir = os.path.join(self.data_dir[0], smpl_name).replace('gap1', gap).replace('.jpg', '.flo')
            if os.path.exists(flow_dir):
                out.append(readFlow(flow_dir, self.resolution))
                fgap.append(gap)
            else:
                flow_dir = os.path.join(self.data_dir[0], smpl_name).replace('gap1', _gap).replace('.jpg', '.flo')
                flow = readFlow(flow_dir, self.resolution)
                out.append(-1. * flow)
                fgap.append(_gap)
        
        out = np.stack(out, 0)
        
        # Read GT segmentations
        if self.data_dir[2] is None:  # No GT provided. Empty array output.
            seg = np.zeros((128, 224, 3))
            seg_gt = processMultiSeg(seg, self.gt_res, self.out_channels) 
        elif self.dataset == 'FBMS': # If temporally sparse GTs are provided, empty arrays will be added to complete the sequence
            seq_name = os.path.dirname(smpl_name)
            samples = sorted(glob.glob(os.path.join(self.data_dir[2], seq_name, '*.png')))
            gt_names = [os.path.join(x.split('/')[-2], x.split('/')[-1]) for x in samples]
            refgt_dir = os.path.join(self.data_dir[2], gt_names[0])
            refseg = readSeg(refgt_dir)
            gt_res = refseg.shape
            if smpl_name.replace('.jpg', '.png') in gt_names: # check if GT exists for particular frame
                gt_dir = os.path.join(self.data_dir[2], smpl_name).replace('.jpg', '.png')
                seg = readSeg(gt_dir) 
            else:
                seg = np.zeros(gt_res)
            seg_gt = processMultiSeg(seg, self.gt_res, self.out_channels, self.dataset)    
        else: # All GTs are provided.
            gt_dir = os.path.join(self.data_dir[2], smpl_name).replace('.jpg', '.png')
            if self.amodal:
                seg_gt = readAmodalSeg(gt_dir.replace('Annotations', 'AmodalAnnotations'), resolution = self.gt_res, out_channels = self.out_channels) 
            else:
                seg = readSeg(gt_dir)  
                seg_gt = processMultiSeg(seg, self.gt_res, self.out_channels, self.dataset)  
                    
        return out, seg_gt

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        outs = []
        seg_gts = []
        smpl_names = self.samples[idx]
             
        for i, name in enumerate(smpl_names):
            out, seg_gt = self.readSingleSmpl(name)
            outs.append(out)
            seg_gts.append(seg_gt)
                
        outs = np.stack(outs, 0)
        seg_gts = np.stack(seg_gts, 0)
        img_dir = [os.path.join(self.data_dir[1], i).replace('.jpg', '.png').split('/')[-2:] for i in smpl_names]

        return outs, seg_gts, img_dir
    

        
