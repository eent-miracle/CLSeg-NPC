from skimage.measure import label
import numpy as np
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.nn.functional import one_hot
import cv2
import random
from scipy.ndimage import distance_transform_edt

def pad_or_trim_prompts(prompt_np, max_prompt_num):
    if prompt_np is None:
        prompt_np = np.zeros((0, 3), dtype=float)
    if prompt_np.shape[0] >= max_prompt_num:
        return prompt_np[:max_prompt_num]
    pad_needed = max_prompt_num - prompt_np.shape[0]
    pad = np.tile(np.array([[0.0, 0.0, -1.0]]), (pad_needed, 1))
    return np.vstack([prompt_np, pad])

def random_sum_to(n, num_terms = None):
    num_terms = (num_terms or random.randint(2, n)) - 1
    a = random.sample(range(1, n), num_terms) + [0, n]
    list.sort(a)
    return [a[i+1] - a[i] for i in range(len(a) - 1)]


def get_first_prompt(mask_cls, resized_size, original_size, dist_thre_ratio=0.1, prompt_num=5, max_prompt_num=8, region_type='random', point_coords_list=None, point_labels_list=None):
    original_w, original_h = original_size # (W, H)

    if point_coords_list is not None and len(point_coords_list) > 0:
        prompt = []
        for idx, point_coords in enumerate(point_coords_list):
            scaled_x = point_coords[0] * (resized_size / original_w) 
            scaled_y = point_coords[1] * (resized_size / original_h)
            scaled_x = np.clip(scaled_x, 0, resized_size - 1)
            scaled_y = np.clip(scaled_y, 0, resized_size - 1)
            lbl = 1
            if point_labels_list is not None and idx < len(point_labels_list):
                lbl = point_labels_list[idx]
            prompt.append([scaled_x, scaled_y, lbl]) 
        prompt = pad_or_trim_prompts(np.array(prompt, dtype=float), max_prompt_num)
        mask_curr = mask_cls
        return prompt, mask_curr

    print("Wrong: point_coords_list is Null。")
    if region_type == 'fixed_pair': 
        X_OFFSET_ORIGINAL = random.uniform(8, 20)
        Y_OFFSET_ORIGINAL = random.uniform(2, 22)
        scaled_x_diff = X_OFFSET_ORIGINAL * (resized_size / original_w)
        scaled_y_diff = Y_OFFSET_ORIGINAL * (resized_size / original_h)
        scaled_x_diff_int = int(round(scaled_x_diff))
        scaled_y_diff_int = int(round(scaled_y_diff))
        margin_x = abs(scaled_x_diff_int) + 55
        margin_y = abs(scaled_y_diff_int) + 55
        fg_coords_y, fg_coords_x = np.where(mask_cls > 0)

        if len(fg_coords_x) == 0:
            empty_prompt = pad_or_trim_prompts(np.zeros((0, 3), dtype=float), max_prompt_num)
            return empty_prompt, np.zeros_like(mask_cls)

        valid_indices = (fg_coords_x >= margin_x) & \
                        (fg_coords_x < resized_size - margin_x) & \
                        (fg_coords_y >= margin_y) & \
                        (fg_coords_y < resized_size - margin_y)

        valid_x = fg_coords_x[valid_indices]
        valid_y = fg_coords_y[valid_indices]

        if len(valid_x) == 0:
            print("Warning: valid_x is Null.")
            random_idx = np.random.randint(0, len(fg_coords_x))
            p1_x = fg_coords_x[random_idx]
            p1_y = fg_coords_y[random_idx]
        else:
            random_idx = np.random.randint(0, len(valid_x))
            p1_x = valid_x[random_idx]
            p1_y = valid_y[random_idx]

        p2_x = np.clip(p1_x + scaled_x_diff_int, 0, resized_size - 1)
        p2_y = np.clip(p1_y + scaled_y_diff_int, 0, resized_size - 1)

        prompt = np.array([
            [p1_x, p1_y, 1],
            [p2_x, p2_y, 1] 
        ], dtype=float)
        prompt = pad_or_trim_prompts(prompt, max_prompt_num)

        mask_curr = mask_cls
        return prompt, mask_curr

    if prompt_num == -1:
        prompt_num = random.randint(1, max_prompt_num)

    label_msk, region_ids = label(mask_cls, connectivity=2, return_num=True)
    ratio_list, regionid_list = [], []
    for region_id in range(1, region_ids + 1):
        binary_msk = np.where(label_msk == region_id, 1, 0)
        sum_mask_cls = np.sum(mask_cls)
        if sum_mask_cls == 0:
            r = 0
        else:
            r = np.sum(binary_msk) / sum_mask_cls
        ratio_list.append(r)
        regionid_list.append(region_id)

    if len(ratio_list) > 0:
        ratio_list, regionid_list = zip(*sorted(zip(ratio_list, regionid_list)))
        regionid_list = regionid_list[::-1]
        if region_type == 'random':
            prompt_num = 1
            regionid_list = [random.choice(regionid_list)]
            prompt_num_each_region = [1]
        elif region_type[:7] == 'largest':
            region_max_num = int(region_type[-1])
            valid_region = min(region_max_num, len(regionid_list))
            if valid_region < prompt_num:
                prompt_num_each_region = random_sum_to(prompt_num, valid_region)
            else:
                prompt_num_each_region = prompt_num * [1]
            regionid_list = regionid_list[:min(valid_region, prompt_num)]

        prompt = []
        mask_curr = np.zeros_like(label_msk)
        cx, cy = 0, 0 
        for reg_id in range(len(regionid_list)):
            binary_msk = np.where(label_msk == regionid_list[reg_id], 1, 0)
            mask_curr = np.logical_or(binary_msk, mask_curr)
            padded_mask = np.uint8(np.pad(binary_msk, ((1, 1), (1, 1)), 'constant'))
            dist_img = cv2.distanceTransform(padded_mask, distanceType=cv2.DIST_L2, maskSize=5).astype(np.float32)[1:-1, 1:-1]
            dist_array = sorted(dist_img.copy().flatten())[::-1]
            dist_array = np.array(dist_array)
            sum_dist_array_gt0 = np.sum(dist_array > 0)
            if sum_dist_array_gt0 == 0:
                dis_thre = 1
            else:
                idx = int(dist_thre_ratio * sum_dist_array_gt0)
                if idx >= len(dist_array):
                    idx = len(dist_array) - 1
                dis_thre = max(dist_array[idx], 1)

            cY, cX = np.where(dist_img >= dis_thre)
            if len(cX) == 0: 
                cY, cX = np.where(binary_msk > 0)
                if len(cX) == 0: continue 

            while prompt_num_each_region[reg_id] > 0:
                random_idx = np.random.randint(0, len(cX))
                cx, cy = int(cX[random_idx]), int(cY[random_idx])
                prompt.append((cx, cy, 1))
                prompt_num_each_region[reg_id] -= 1

        while len(prompt) < max_prompt_num:
            if not prompt: 
                prompt.append((0, 0, -1))
                cx, cy = 0, 0
            else:
                prompt.append((cx, cy, 1))
    else: 
        prompt = [(0, 0, -1)]
        mask_curr = np.zeros_like(label_msk)
        while len(prompt) < max_prompt_num: 
            prompt.append((0, 0, -1))

    prompt = pad_or_trim_prompts(np.array(prompt, dtype=float), max_prompt_num)
    mask_curr = np.array(mask_curr, dtype=int)
    return prompt, mask_curr

def get_top_boxes(mask_cls,dist_thre_ratio=0.1,region_max_num=5,region_type='largest_5'):
    label_msk, region_ids = label(mask_cls, connectivity=2, return_num=True)
    ratio_list, regionid_list = [], []
    for region_id in range(1, region_ids+1):
        binary_msk = np.where(label_msk==region_id, 1, 0)
        sum_mask_cls = np.sum(mask_cls)
        if sum_mask_cls == 0:
            r = 0
        else:
            r = np.sum(binary_msk) / sum_mask_cls
        ratio_list.append(r)
        regionid_list.append(region_id)
        
    if len(ratio_list)>0:
        ratio_list, regionid_list = zip(*sorted(zip(ratio_list, regionid_list)))
        regionid_list = regionid_list[::-1]

        if region_type == 'random':
            region_max_num = 1
            regionid_list = [random.choice(regionid_list)] 
        elif region_type[:7] == 'largest':
            region_max_num = int(region_type[-1])
            regionid_list = regionid_list[:min(region_max_num,len(regionid_list))]

        prompt = []
        mask_curr = np.zeros_like(label_msk)
        
        box = [0,0,0,0] 
        
        for reg_id in range(len(regionid_list)):
            binary_msk = np.where(label_msk==regionid_list[reg_id], 1, 0)
            mask_curr = np.logical_or(binary_msk,mask_curr)
            box = MaskToBoxSimple(binary_msk,dist_thre_ratio)
            prompt.append(box)

        while len(prompt)<region_max_num: 
            prompt.append(box)
        prompt = np.array(prompt) 
        mask_curr = np.array(mask_curr,dtype=int)
    else:
        prompt = [[0,0,0,0]]
        mask_curr = np.zeros_like(label_msk)
        while len(prompt)<region_max_num:
            prompt.append(prompt[0])
    return prompt,mask_curr
    
def MaskToBoxSimple(mask,random_thre=0.1):
    mask = mask.squeeze()
    
    if np.sum(mask) == 0:
        return [0, 0, 0, 0]
        
    y_max,x_max = mask.shape[0],mask.shape[1]
    
    row, col = np.argwhere(mask).T
    y0,x0 = row.min(),col.min()
    y1,x1 = row.max(),col.max()
    
    y_thre = (y1-y0)*random_thre
    x_thre = (x1-x0)*random_thre
    
    x0 = max(0,x0-x_thre*random.random())
    x1 = min(x_max,x1+x_thre*random.random())
    
    y0 = max(0,y0-y_thre*random.random())
    y1 = min(y_max,y1+y_thre*random.random())
    
    return [x0,y0,x1,y1]
