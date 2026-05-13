#!/usr/bin/env python
# coding: utf-8

from models.sam import SamPredictor, sam_model_registry
from models.sam.utils.transforms import ResizeLongestSide
from skimage.measure import label as sk_label 
from models.sam_LoRa import LoRA_Sam
import numpy as np
import os
import torch
from torch import nn
import torch.optim as optim
import torchvision
from torchvision import datasets
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torch.autograd import Variable
import matplotlib.pyplot as plt
import copy
from utils.dataset import Public_dataset
import torch.nn.functional as F
from torch.nn.functional import one_hot
from pathlib import Path
from tqdm import tqdm
from utils.losses import DiceLoss
from utils.dsc import dice_coeff_multi_class
import cv2 
import monai
import cfg
import json
import random

# Use the arguments
args = cfg.parse_args()

def compute_iou_targets(pred_logits: torch.Tensor, gt_masks: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    gt = (gt_masks > 0.5).float()
    pred_bin = (torch.sigmoid(pred_logits) > 0.5).float()
    gt_exp = gt.expand_as(pred_bin)
    inter = (pred_bin * gt_exp).sum(dim=(2, 3))
    union = (pred_bin + gt_exp - pred_bin * gt_exp).sum(dim=(2, 3))
    return inter / (union + eps)

def cleanup_mask_based_on_points(mask, pos_points_np):
    labeled_mask, num_labels = sk_label(mask, connectivity=2, background=0, return_num=True)
    
    final_mask_np = np.zeros_like(mask, dtype=np.uint8)
    if num_labels == 0: 
        return final_mask_np 

    point_coords_int = np.round(pos_points_np).astype(int)
    target_labels = set()
    
    for point in point_coords_int:
        x, y = point[0], point[1]
        if 0 <= y < labeled_mask.shape[0] and 0 <= x < labeled_mask.shape[1]: 
            label_at_point = labeled_mask[y, x] 
            if label_at_point > 0: 
                target_labels.add(label_at_point)
    
    if target_labels:
        is_in_target_region = np.isin(labeled_mask, list(target_labels))
        raw_target_mask = np.where(is_in_target_region, mask, 0).astype(np.uint8)

        kernel_open_size = 5
        open_iterations = 2
        kernel_open = np.ones((kernel_open_size, kernel_open_size), np.uint8)
        opened_mask = cv2.morphologyEx( (raw_target_mask > 0).astype(np.uint8), cv2.MORPH_OPEN, kernel_open, iterations=open_iterations)

        kernel_close_size = 7
        close_iterations = 1
        kernel_close = np.ones((kernel_close_size, kernel_close_size), np.uint8)
        closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel_close, iterations=close_iterations)
        
        final_mask_np = np.where(closed_mask > 0, raw_target_mask, 0).astype(np.uint8)
    
    return final_mask_np

def train_model(trainloader,valloader,dir_checkpoint,epochs):
    if args.if_warmup:
        b_lr = args.lr / args.warmup_period
    else:
        b_lr = args.lr
    
    sam = sam_model_registry[args.arch](args,checkpoint=os.path.join(args.sam_ckpt),num_classes=args.num_cls)
    if args.finetune_type == 'adapter':
        for n, value in sam.named_parameters():
            if "Adapter" not in n: # only update parameters in adapter
                value.requires_grad = False
        print('if update encoder:',args.if_update_encoder)
        print('if image encoder adapter:',args.if_encoder_adapter)
        print('if mask decoder adapter:',args.if_mask_decoder_adapter)
        if args.if_encoder_adapter:
            print('added adapter layers:',args.encoder_adapter_depths)
    elif args.finetune_type == 'vanilla' and args.if_update_encoder==False:   
        print('if update encoder:',args.if_update_encoder)
        for n, value in sam.image_encoder.named_parameters():
            value.requires_grad = False
    elif args.finetune_type == 'lora':
        print('if update encoder:',args.if_update_encoder)
        print('if image encoder lora:',args.if_encoder_lora_layer)
        print('if mask decoder lora:',args.if_decoder_lora_layer)
        sam = LoRA_Sam(args,sam,r=4).sam

    sam.to('cuda')
        
    optimizer = optim.AdamW(sam.parameters(), lr=b_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.1, amsgrad=False)
    optimizer.zero_grad()
    # LR is managed manually: linear warmup followed by polynomial decay (1 - t/T)^0.9
    criterion1 = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, to_onehot_y=False, reduction='mean')
    criterion2 = nn.BCEWithLogitsLoss()
    iou_loss_weight = getattr(args, "iou_loss_weight", 1.0)
    
    iter_num = 0
    max_iterations = epochs * len(trainloader) 
    writer = SummaryWriter(dir_checkpoint + '/log')
    
    pbar = tqdm(range(epochs))
    val_largest_dsc = 0
    last_update_epoch = 0
    
    MODEL_INPUT_SIZE = 1024.0

    for epoch in pbar:
        sam.train()
        switch_epoch = int(epochs * args.dropout_switch_ratio)
        if epoch < switch_epoch:
            point_drop_prob = args.point_drop_high if args.point_drop_high is not None else args.point_drop_prob
            box_drop_prob = args.box_drop_high if args.box_drop_high is not None else args.box_drop_prob
        else:
            point_drop_prob = args.point_drop_low if args.point_drop_low is not None else args.point_drop_prob
            box_drop_prob = args.box_drop_low if args.box_drop_low is not None else args.box_drop_prob

        train_loss = 0
        for i,data in enumerate(tqdm(trainloader)):
            imgs = data['image'].cuda()
            msks = torchvision.transforms.Resize((args.out_size,args.out_size))(data['mask'])
            msks = msks.cuda()
            if args.with_boxes:
                boxes = data['boxes'].cuda() 
            else:
                boxes = None
            points = data['point_coords'].cuda()
            point_labels = data['point_labels'].cuda()

            if args.with_points and point_drop_prob > 0.0:
                orig_points = points
                orig_labels = point_labels
                drop_mask = (torch.rand_like(point_labels) < point_drop_prob) & (point_labels != -1)
                if drop_mask.any():
                    points = points.clone()
                    point_labels = point_labels.clone()
                    points[drop_mask] = 0
                    point_labels[drop_mask] = -1
                if args.keep_one_prompt and (boxes is None):
                    for b in range(point_labels.size(0)):
                        if (point_labels[b] != -1).any():
                            continue
                        idx = (orig_labels[b] != -1).nonzero(as_tuple=False)
                        if idx.numel() > 0:
                            j = idx[0].item()
                            point_labels[b, j] = orig_labels[b, j]
                            points[b, j] = orig_points[b, j]

            if args.with_boxes and box_drop_prob > 0.0 and boxes is not None:
                if random.random() < box_drop_prob:
                    boxes = None

            if args.if_update_encoder: # True
                img_emb = sam.image_encoder(imgs)
            else:
                with torch.no_grad():
                    img_emb = sam.image_encoder(imgs)
            
            if args.with_points:
                sparse_emb, dense_emb = sam.prompt_encoder(
                    points=(points, point_labels),
                    boxes=boxes,
                    masks=None,
                )
            else:
                sparse_emb, dense_emb = sam.prompt_encoder(
                    points=None,
                    boxes=boxes,
                    masks=None,
                )                

            pred, iou_predictions = sam.mask_decoder(
                            image_embeddings=img_emb,
                            image_pe=sam.prompt_encoder.get_dense_pe(), 
                            sparse_prompt_embeddings=sparse_emb,
                            dense_prompt_embeddings=dense_emb, 
                            multimask_output=True,
                          )
            best_idx = torch.argmax(iou_predictions, dim=1)
            pred_best = pred[torch.arange(pred.size(0), device=pred.device), best_idx].unsqueeze(1)
            iou_targets = compute_iou_targets(pred, msks.float())
            iou_pred = torch.sigmoid(iou_predictions)
            loss_iou = F.mse_loss(iou_pred, iou_targets)
            loss_dice = criterion1(pred_best, msks.float())
            loss_bce = criterion2(pred_best, msks.float())
            loss = loss_dice + loss_bce + iou_loss_weight * loss_iou
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            if args.if_warmup and iter_num < args.warmup_period:
                lr_ = args.lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            else:
                if args.if_warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                    lr_ = args.lr * (1.0 - shift_iter / max_iterations) ** 0.9  # learning rate adjustment depends on the max iterations
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_
                else:
                    lr_ = args.lr

            train_loss += loss.item()
            iter_num+=1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_bce', loss_bce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            writer.add_scalar('info/loss_iou', loss_iou, iter_num)

        train_loss /= (i+1)
        pbar.set_description('Epoch num {}| train loss {} \n'.format(epoch,train_loss))

        if epoch%2==0:
            eval_loss=0
            dsc = 0
            sam.eval()
            with torch.no_grad():
                for i,data in enumerate(tqdm(valloader)):
                    imgs = data['image'].cuda()
                    msks = torchvision.transforms.Resize((args.out_size,args.out_size))(data['mask'])
                    msks = msks.cuda() # GT Mask [B, 1, 256, 256]    
                    points = data['point_coords'].cuda() # Points [B, N, 2] (scaled to 1024)
                    point_labels = data['point_labels'].cuda() # Labels [B, N]
                    if args.with_boxes:
                        boxes = data['boxes'].cuda() 
                    else:
                        boxes = None

                    img_emb= sam.image_encoder(imgs)
                    if args.with_points:
                        sparse_emb, dense_emb = sam.prompt_encoder(
                            points=(points, point_labels), 
                            boxes=boxes,                    
                            masks=None,
                        )
                    else:
                        sparse_emb, dense_emb = sam.prompt_encoder(
                            points=None,
                            boxes=boxes,
                            masks=None,
                        )
                    pred, iou_predictions = sam.mask_decoder(
                                    image_embeddings=img_emb,
                                    image_pe=sam.prompt_encoder.get_dense_pe(), 
                                    sparse_prompt_embeddings=sparse_emb,
                                    dense_prompt_embeddings=dense_emb, 
                                    multimask_output=True,
                                  )
                    best_idx = torch.argmax(iou_predictions, dim=1)
                    pred_best = pred[torch.arange(pred.size(0), device=pred.device), best_idx].unsqueeze(1)
                    iou_targets = compute_iou_targets(pred, msks.float())
                    iou_pred = torch.sigmoid(iou_predictions)
                    loss_iou = F.mse_loss(iou_pred, iou_targets)
                    loss = criterion1(pred_best, msks.float()) + criterion2(pred_best, msks.float()) + iou_loss_weight * loss_iou
                    eval_loss += loss.item()
                    
                    pred_mask = (torch.sigmoid(pred_best) > 0.5).long().cpu().squeeze(1)
                    dsc_batch = dice_coeff_multi_class(pred_mask, torch.squeeze(msks.long(),1).cpu().long(), args.num_cls)
                    dsc+=dsc_batch

            eval_loss /= (i+1)
            dsc /= (i+1)
            
            writer.add_scalar('eval/loss', eval_loss, epoch)
            writer.add_scalar('eval/dice', dsc, epoch) 
            
            print('Eval Epoch num {} | val loss {} | dsc {} \n'.format(epoch,eval_loss,dsc))
            if dsc>val_largest_dsc:
                val_largest_dsc = dsc
                last_update_epoch = epoch
                print('largest DSC (cleaned) now: {}'.format(dsc))
                torch.save(sam.state_dict(),dir_checkpoint + '/checkpoint_best.pth')
            elif (epoch-last_update_epoch)>200:
                print('Training finished###########')
                break
    writer.close()
            
            
if __name__ == "__main__":
    dataset_name = args.dataset_name
    print('train dataset: {}'.format(dataset_name)) 
    train_img_list = args.train_img_list
    val_img_list = args.val_img_list
    
    num_workers = 8
    if_vis = True
    Path(args.dir_checkpoint).mkdir(parents=True,exist_ok = True)
    path_to_json = os.path.join(args.dir_checkpoint, "args.json")
    args_dict = vars(args)
    with open(path_to_json, 'w') as json_file:
        json.dump(args_dict, json_file, indent=4)

    train_dataset = Public_dataset(args,args.img_folder, args.mask_folder, args.point_folder, train_img_list,phase='train',targets=[args.targets],normalize_type='sam',if_prompt=True,prompt_type='point', region_type=args.region_type)
    eval_dataset = Public_dataset(args,args.img_folder, args.mask_folder, args.point_folder, val_img_list,phase='val',targets=[args.targets],normalize_type='sam',if_prompt=True,prompt_type='point',region_type=args.region_type)
    
    trainloader = DataLoader(train_dataset, batch_size=args.b, shuffle=True, num_workers=num_workers)
    valloader = DataLoader(eval_dataset, batch_size=args.b, shuffle=False, num_workers=num_workers)

    train_model(trainloader,valloader,args.dir_checkpoint,args.epochs)
