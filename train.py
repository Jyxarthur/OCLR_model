import os
import time
import einops
import numpy as np
import utils as ut
import config as cg
import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter
from argparse import ArgumentParser
from models.oclr import OCLR
from eval import eval


def main(args):
    lr = args.lr
    batch_size = args.batch_size
    eval_freq = args.eval_freq
    optim_freq = args.optim_freq
    warmup_it = args.warmup_steps
    decay_step = args.decay_steps
    num_it = args.num_train_steps
    resume_path = args.resume_path
    frames = args.frames
    args.resolution = (128, 224)
    
    
    # training weight
    weight_amodal_bce = 1.0
    weight_amodal_bound = 0.2
    weight_modal = 0.1  # or 0.
    weight_order = 0.05
    
    
    # setup log and model paths,
    [logPath, modelPath] = cg.setup_path(args)
    
    # initialize tensorboard
    writer = SummaryWriter(logdir=logPath)
    log_freq = 50  # reporting frequency to tensorboard
    

    # initialize dataloader 
    trn_dataset, val_dataset, resolution, in_channels, out_channels = cg.setup_dataset(args)
    trn_loader = ut.FastDataLoader(
        trn_dataset, num_workers=8, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
    val_loader = ut.FastDataLoader(
        val_dataset, num_workers=8, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=False)
    
    # initialize model and optimiser
    it = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = OCLR(in_channels, out_channels, num_query = args.queries)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if resume_path:
        print('resuming from checkpoint')
        model_dict = model.state_dict()
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        it = checkpoint['iteration']
        loss = checkpoint['loss']
    else:
        print('training from scratch')

    
    # main training iterations
    print('======> start training {}, use {}.'.format(args.dataset, device))
    timestart = time.time()
    while it < num_it:
        for i, sample in enumerate(trn_loader):
            flow, gt_am, _ = sample
            gt_am = gt_am.float().to(device)   # "_am" represents amodal masks.
            gt_m = ut.find_recon_mask(gt_am, torch.from_numpy(np.arange(gt_am.size()[2])).long())   # "_m" represents modal masks.  
            flow = flow.float().to(device)
            flow = einops.rearrange(flow, 'b t g c h w -> b t (g c) h w')

            mask_am_raw, order_raw = model(flow)
            
            # Ignore unstable outputs appearing very occasionally
            mask_am_raw = torch.nan_to_num(mask_am_raw, nan = 0., posinf = 0., neginf = 0.) 
            order_raw = torch.nan_to_num(order_raw, nan = 0., posinf = 0., neginf = 0.)
            
            mask_am_raw, order_raw = ut.hungarian_matcher(mask_am_raw, gt_am, [mask_am_raw, order_raw]) # hungarian matching

            # amodal losses
            loss_amodal_bce = weight_amodal_bce * ut.criterion_amodal_bce(mask_am_raw, gt_am)
            loss_amodal_bound = weight_amodal_bound * ut.criterion_amodal_bound(mask_am_raw, gt_am)
            # modal loss (newly introduced compared to the original paper --- can be ignored by setting to zero / very small weight, mainly used to train the layer ordering head)
            mask_am = mask_am_raw.sigmoid()  
            mask_m = ut.amodal_to_modal_soft(mask_am, order_raw) 
            loss_modal = weight_modal * ut.criterion_modal(mask_m, gt_m)
            # order loss
            loss_order = weight_order * ut.criterion_order(order_raw, gt_am)

            # total loss
            loss = (loss_amodal_bce + loss_amodal_bound) + loss_order + loss_modal
            loss = loss / optim_freq
            loss.backward()
                  
            # training set report to tensorboard
            if it % log_freq == 0:
                writer.add_scalar('Loss/total', loss.detach().cpu().numpy(), it)
                writer.add_scalar('Loss/amodal_bce', loss_amodal_bce.detach().cpu().numpy(), it)
                writer.add_scalar('Loss/amodal_bound', loss_amodal_bound.detach().cpu().numpy(), it)
                writer.add_scalar('Loss/order', loss_order.detach().cpu().numpy(), it)
                writer.add_scalar('Loss/modal', loss_modal.detach().cpu().numpy(), it)
                ious_m = []
                ious_am = []
                for i in range(mask_m.size()[0]):
                    ious_m.extend(ut.hungarian_iou(mask_m[i], gt_m[i]))
                    ious_am.extend(ut.hungarian_iou(mask_am[i], gt_am[i]))
                iou_m = np.mean(np.array(ious_m))
                writer.add_scalar('IOU/train_modal', iou_m, it)
                iou_am = np.mean(np.array(ious_am))
                writer.add_scalar('IOU/train_amodal', iou_am, it)
            
            # validation set report to tensorboard & saving ckpts
            if it % eval_freq == 0:
                meaniou_m, meaniou_am = eval(val_loader, model, device, args = args)
                writer.add_scalar('IOU/val_modal', meaniou_m, it)
                writer.add_scalar('IOU/val_amodal', meaniou_am, it)
                
                filename = os.path.join(modelPath, 'ckpt_{}-(modal_{})-(amodal_{}).pth'.format(it, np.round(meaniou_m, 3), np.round(meaniou_am, 3)))
                torch.save({
                    'iteration': it,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }, filename)
                
            # gradient value clipping
            nn.utils.clip_grad_value_(model.parameters(), clip_value=2.0)
                
            # optimiser updates
            if it % optim_freq == 0:
                optimizer.step()
                optimizer.zero_grad()  
            if it < warmup_it:  # warmup steps
                ut.set_learning_rate(optimizer, lr * it / warmup_it)
            if it % decay_step == 0 and it // decay_step <= 9:  # decaying steps (lr divided by 2)
                ut.set_learning_rate(optimizer, lr * (0.5 ** (it // decay_step)))        
            
            print('iteration {},'.format(it),
                  'time {:.01f}s,'.format(time.time() - timestart),
                  'loss {}.'.format(loss.detach().cpu().numpy()))
            it += 1
            timestart = time.time()
                
                
                
if __name__ == "__main__":
    parser = ArgumentParser()
    # training settings
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--num_train_steps', type=int, default=600000)
    parser.add_argument('--warmup_steps', type=int, default=40000)
    parser.add_argument('--decay_steps', type=int, default=80000)
    parser.add_argument('--eval_freq', type=int, default=20000)
    parser.add_argument('--optim_freq', type=int, default=1)
    
    # input settings
    parser.add_argument('--queries', type=int, default=3)
    parser.add_argument('--dataset', type=str, default='Syn', choices=['Syn', 'DAVIS17m', 'DAVIS16', 'Segtrack', 'FBMS', 'MoCA'])
    parser.add_argument('--gaps', type=str, default='1,-1')  # Two flow gaps inputs, input string should not include space in-between.
    parser.add_argument('--frames', type=int, default=30)
    
    # paths
    parser.add_argument('--resume_path', type=str, default=None)
    parser.add_argument('--save_path', type=str, default=None)  # For training, keep this as None
    
    args = parser.parse_args()
    args.inference = False
    main(args)
