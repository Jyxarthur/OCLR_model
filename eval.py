import sys
import torch
import einops
import numpy as np
import utils as ut
import config as cg

from argparse import ArgumentParser
from models.oclr import OCLR

def main(args):
    batch_size = args.batch_size
    resume_path = args.resume_path
    args.resolution = (128, 224)
    dataset = args.dataset
    if dataset in ['Segtrack', 'FBMS', 'DAVIS17m']:
        batch_size = 1  # since images are of different sizes
        
    # setup log and model path
    [logPath, modelPath] = cg.setup_path(args)
    _, val_dataset, resolution, in_channels, out_channels = cg.setup_dataset(args)
    val_loader = ut.FastDataLoader(
        val_dataset, num_workers=8, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=False)
        
    # initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = OCLR(in_channels, out_channels, num_query = args.queries)
    model.to(device)

    if resume_path:
        print('resuming from checkpoint')
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    else:
        print('no checkpoint found')
        sys.exit(0)

    print('======> start inference {}, use {}.'.format(args.dataset, device))
    #inference / evaluate on validation set
    print(eval(val_loader, model, device, Eval = True, args = args))


def eval(val_loader, model, device, Eval=False, args = None):
    save_path = args.save_path
    dataset = args.dataset

    with torch.no_grad():
        avgmeter = ut.AverageMeter()
        
        print(' --> running inference')
        for idx, val_sample in enumerate(val_loader): 
            if idx % 10 == 0:
                print(' - evaluating iteration ' + str(idx))
            flow, gt_am, meta = val_sample
            meta = np.array(meta) 
            categories, indices = meta[0, 0, :], meta[:, 1, :]
            gt_am = gt_am.float().to(device)
            gt_m = ut.find_recon_mask(gt_am, torch.from_numpy(np.arange(gt_am.size()[2])).long()).detach()
            t = flow.size()[1]
            flow = flow.float().to(device)
            flow = einops.rearrange(flow, 'b t g c h w -> b t (g c) h w')
    
            mask_am_raw, order_raw = model(flow)   # "_raw" means the range of value is not constraint. i.e. not perform sigmoid operation yet.
            
            # Note here masks and pred. orders are not hungarian matched
            mask_am = mask_am_raw.sigmoid()
            mask_m = ut.amodal_to_modal_hard(mask_am, order_raw)
            
            if dataset in ['Segtrack', 'FBMS']:
                mask_m[:, :, 0] = torch.clamp(torch.sum(mask_m, 2), 0., 1.)
                mask_m[:, :, 1:] = 0 *  mask_m[:, :, 1:]             
            
            for i in range(mask_m.size()[0]):
                category = categories[i]
                index = indices[:, i]
                filenames = index.tolist()
                iou_am = ut.hungarian_iou(mask_am[i], gt_am[i])   
                iou_m = ut.hungarian_iou(mask_m[i], gt_m[i])
                for j, filename in enumerate(filenames):
                    avgmeter.update('iou_am', category, filename, iou_am[j])
                    avgmeter.update('iou_m', category, filename, iou_m[j])
                if Eval and save_path:
                    # Before saving, Hungarian matching according to IoU
                    mask_m_hung, mask_am_hung = ut.hungarian_matcher_iou(mask_m, gt_m, [mask_m, mask_am])
                    _, _, _, H, W = gt_m.size()
                    ut.save_vis_results(mask_m_hung[i:i+1], mask_am_hung[i:i+1], (H, W), dataset, category, filenames, save_path)
        mean_iou_m, mean_iou_am = avgmeter.summary(dataset)
    return mean_iou_m, mean_iou_am
 
 
if __name__ == "__main__":
    parser = ArgumentParser()
    # inference settings
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--queries', type=int, default=3)
    # input settings
    parser.add_argument('--dataset', type=str, default='Syn', choices=['Syn', 'DAVIS17m', 'DAVIS16', 'Segtrack', 'FBMS', 'MoCA'])
    parser.add_argument('--gaps', type=str, default='1,-1')  # Two flow gaps inputs, input string should not include space in-between.
    parser.add_argument('--frames', type=int, default=5)
    # paths
    parser.add_argument('--resume_path', type=str, default=None)
    parser.add_argument('--save_path', type=str, default = None)
    
    args = parser.parse_args()
    args.inference = True
    main(args)

