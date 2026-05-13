import os, torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import random
import pickle
from utils.funcs import *
from torchvision.transforms import InterpolationMode
import json
from pathlib import Path

def parse_point_json(json_path):
    point_coords_list, point_labels_list, bbox = [], [], None
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    if 'points' in json_data:
        for idx, point in enumerate(json_data.get('points', [])):
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            lbl = 1
            if 'point_labels' in json_data and idx < len(json_data['point_labels']):
                lbl = json_data['point_labels'][idx]
            point_coords_list.append(point[:2])
            point_labels_list.append(lbl)
        bbox = json_data.get('bbox', None)
    elif 'shapes' in json_data:
        for shape in json_data['shapes']:
            for point in shape['points']:
                point_coords_list.append(point)
                point_labels_list.append(1)
        bbox = json_data.get('bbox', None)

    return point_coords_list, point_labels_list, bbox

def pad_or_trim_boxes(box_np, max_num=1):
    if box_np is None:
        box_np = np.zeros((0, 4), dtype=float)
    if box_np.shape[0] >= max_num:
        return box_np[:max_num]
    pad_needed = max_num - box_np.shape[0]
    pad = np.zeros((pad_needed, 4), dtype=float)
    return np.vstack([box_np, pad])

class Public_dataset(Dataset):
    def __init__(self,args, img_folder, mask_folder, point_folder, img_list,phase='train',sample_num=50,channel_num=1,normalize_type='sam',crop=False,crop_size=1024,targets=['femur','hip'],part_list=['all'],cls=-1,if_prompt=True,prompt_type='point',region_type='largest_3',label_mapping=None,if_spatial=True,delete_empty_masks=True):
        '''
        target: 'combine_all': combine all the targets into binary segmentation
                'multi_all': keep all targets as multi-cls segmentation
                f'{one_target_name}': segmentation specific one type of target, such as 'hip'
        
        normalzie_type: 'sam' or 'medsam', if sam, using transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]); if medsam, using [0,1] normalize
        cls: the target cls for segmentation
        prompt_type: point or box
        if_patial: if add spatial transformations or not
        
        '''
        super(Public_dataset, self).__init__()
        self.args = args
        self.img_folder = img_folder
        self.mask_folder = mask_folder
        self.point_folder = point_folder
        self.crop = crop
        self.crop_size = crop_size
        self.phase = phase
        self.normalize_type = normalize_type
        self.targets = targets
        self.part_list = part_list
        self.cls = cls
        self.delete_empty_masks = delete_empty_masks
        self.if_prompt = if_prompt
        self.prompt_type = prompt_type
        self.region_type = region_type
        self.label_dic = {}
        self.data_list = []
        self.label_mapping = label_mapping
        self.load_label_mapping()
        self.load_data_list(img_list)
        self.if_spatial = if_spatial
        self.setup_transformations()

    def load_label_mapping(self):
        if self.label_mapping:
            with open(self.label_mapping, 'rb') as handle:
                self.segment_names_to_labels = pickle.load(handle)
            self.label_dic = {seg[1]: seg[0] for seg in self.segment_names_to_labels}
            self.label_name_list = [seg[0] for seg in self.segment_names_to_labels]
            print(self.label_dic)
        else:
            self.segment_names_to_labels = {}
            self.label_dic = {value: 'all' for value in range(1, 256)}
        
    def load_data_list(self, img_list):
        with open(img_list, 'r', encoding='utf-8-sig') as file:
            lines = file.read().strip().split('\n')
        for line in lines:
            if len(line.split(',')) == 2:
                img_path, mask_path = line.split(',')
                if self.point_folder:
                    candidate = Path(img_path.strip()).stem + '_prompt.json'
                    if os.path.exists(os.path.join(self.point_folder, candidate)):
                        line = ','.join([img_path, mask_path, candidate])
            elif len(line.split(',')) == 3:
                img_path, mask_path, point_path = line.split(',')
                point_path = point_path.strip()
            mask_path = mask_path.strip()
            if mask_path.startswith('/'):
                mask_path = mask_path[1:]
            
            # skip if mask file is missing
            full_mask_path = os.path.join(self.mask_folder, mask_path)
            if not os.path.exists(full_mask_path):
                print(f"Waring: no masks {full_mask_path}, skip...")
                continue
                
            msk = Image.open(full_mask_path).convert('L')
            if self.should_keep(msk, mask_path):
                self.data_list.append(line)

        print(f'Filtered data list to {len(self.data_list)} entries.')

    def should_keep(self, msk, mask_path):
        if self.delete_empty_masks:
            mask_array = np.array(msk, dtype=int)
            if 'combine_all' in self.targets:
                return np.any(mask_array > 0)
            elif 'multi_all' in self.targets:
                return np.any(mask_array > 0)
            elif any(target in self.targets for target in self.segment_names_to_labels):
                target_classes = [self.segment_names_to_labels[target][1] for target in self.targets if target in self.segment_names_to_labels]
                return any(mask_array == cls for cls in target_classes)
            elif self.cls > 0:
                return np.any(mask_array == self.cls)
            if self.part_list[0] != 'all':
                return any(part in mask_path for part in self.part_list)
            return False
        else:
            return True

    def setup_transformations(self):
        if self.phase =='train':
            transformations = [transforms.RandomEqualize(p=0.1),
                               transforms.ColorJitter(brightness=0.3, contrast=0.3,saturation=0.3,hue=0.3),
                               ]
            if self.if_spatial:
                self.transform_spatial = transforms.Compose([transforms.RandomResizedCrop(self.crop_size, scale=(0.5, 1.5), interpolation=InterpolationMode.NEAREST),
                                                             transforms.RandomRotation(45, interpolation=InterpolationMode.NEAREST)])
        else:
            transformations = []
        transformations.append(transforms.ToTensor())
        if self.normalize_type == 'sam':
            transformations.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        elif self.normalize_type == 'medsam':
            transformations.append(transforms.Lambda(lambda x: (x - torch.min(x)) / (torch.max(x) - torch.min(x))))
        self.transform_img = transforms.Compose(transformations)

    def __len__(self):
        return len(self.data_list)

    def need_save(self):
        check_data = self.data_list[0]
        return len(check_data.split(',')) == 3

    def __getitem__(self, index):
        data = self.data_list[index]
        point_coords_list, point_labels_list, bbox_from_json = None, None, None
        if len(data.split(',')) == 2:
            img_path, mask_path = data.split(',')
        elif len(data.split(',')) == 3:
            img_path, mask_path, point_path = data.split(',')
            full_json_path = os.path.join(self.point_folder, point_path.strip())
            try:
                point_coords_list, point_labels_list, bbox_from_json = parse_point_json(full_json_path)
            except FileNotFoundError:
                print(f"Warning: no JSON {full_json_path}")
                point_coords_list, point_labels_list, bbox_from_json = None, None, None
        
        if mask_path.startswith('/'):
            mask_path = mask_path[1:]
        
        img_pil = Image.open(os.path.join(self.img_folder, img_path.strip())).convert('RGB')
        msk_pil = Image.open(os.path.join(self.mask_folder, mask_path.strip())).convert('L')

        original_size = img_pil.size 

        img = transforms.Resize((self.args.image_size,self.args.image_size))(img_pil)
        msk = transforms.Resize((self.args.image_size,self.args.image_size),InterpolationMode.NEAREST)(msk_pil)
        
        img, msk = self.apply_transformations(img, msk)

        if 'combine_all' in self.targets: # combine all targets as single target
            msk = (msk > 0).int()
        elif 'multi_all' in self.targets:
            msk = msk.int()
        elif self.cls > 0:
            msk = (msk == self.cls).int()
            
        return self.prepare_output(img, msk, img_path, mask_path, original_size, point_coords_list, point_labels_list, bbox_from_json)

    def apply_transformations(self, img, msk):
        if self.crop:
            img, msk = self.apply_crop(img, msk)
        
        img_tensor = self.transform_img(img) 
        msk_tensor = torch.tensor(np.array(msk, dtype=int), dtype=torch.long) 

        if self.phase=='train' and self.if_spatial:
            mask_cls = msk_tensor.numpy() 
            mask_cls = np.repeat(mask_cls[np.newaxis,:, :], 3, axis=0) 
            
            both_targets = torch.stack((img_tensor, torch.tensor(mask_cls).float()), 0)
            
            transformed_targets = self.transform_spatial(both_targets) # [2, 3, H, W]
            
            img_tensor = transformed_targets[0] # [3, H, W]
            mask_cls = np.array(transformed_targets[1][0].detach(),dtype=int) 
            msk_tensor = torch.tensor(mask_cls) # [H, W]
            
        return img_tensor, msk_tensor

    def apply_crop(self, img, msk):
        t, l, h, w = transforms.RandomCrop.get_params(img, (self.crop_size, self.crop_size))
        img = transforms.functional.crop(img, t, l, h, w)
        msk = transforms.functional.crop(msk, t, l, h, w)
        return img, msk

    def prepare_output(self, img, msk, img_path, mask_path, original_size, point_coords_list, point_labels_list, bbox_from_json):
        if len(msk.shape) == 2:
            msk = torch.unsqueeze(msk, 0) 
        
        original_mask = msk.clone() 
        output = {'image': img, 'mask': original_mask, 'img_name': img_path}

        if self.if_prompt:
            if self.prompt_type == 'point':
                prompt, mask_now = get_first_prompt(
                    msk.cpu().numpy()[0],  # msk (H, W) as numpy
                    resized_size=self.args.image_size,
                    original_size=original_size,
                    region_type=self.region_type,
                    point_coords_list=point_coords_list, 
                    point_labels_list=point_labels_list
                )
                
                pc = torch.tensor(prompt[:, :2], dtype=torch.float32).clone()
                pl = torch.tensor(prompt[:, -1], dtype=torch.float32).clone()
                output.update({'point_coords': pc, 'point_labels': pl})

                if bbox_from_json is not None:
                    ow, oh = original_size
                    scale_x = self.args.image_size / ow
                    scale_y = self.args.image_size / oh
                    x0, y0, x1, y1 = bbox_from_json
                    max_coord = self.args.image_size - 1
                    scaled_box = [[
                        max(0, min(max_coord, x0 * scale_x)),
                        max(0, min(max_coord, y0 * scale_y)),
                        max(0, min(max_coord, x1 * scale_x)),
                        max(0, min(max_coord, y1 * scale_y)),
                    ]]
                    box_np = pad_or_trim_boxes(np.array(scaled_box, dtype=float), max_num=1)
                    box = torch.tensor(box_np, dtype=torch.float32).clone()
                else:
                    box_prompt, _ = get_top_boxes(msk.cpu().numpy()[0], region_type='largest_1')
                    box_np = pad_or_trim_boxes(np.array(box_prompt, dtype=float), max_num=1)
                    box = torch.tensor(box_np, dtype=torch.float32).clone()
                output.update({'boxes': box})
            
            elif self.prompt_type == 'box':
                prompt, mask_now = get_top_boxes(msk.cpu().numpy()[0], region_type=self.region_type)
                box = torch.tensor(prompt, dtype=torch.float)
                output.update({'boxes': box})
            
            elif self.prompt_type == 'hybrid':
                point_prompt, _ = get_first_prompt(msk[0].numpy(), resized_size=self.args.image_size, original_size=original_size, region_type=self.region_type)
                box_prompt, _ = get_top_boxes(msk.numpy()[0], region_type=self.region_type)
                pc = torch.tensor(point_prompt[:, :2], dtype=torch.float)
                pl = torch.tensor(point_prompt[:, -1], dtype=torch.float)
                box = torch.tensor(box_prompt, dtype=torch.float)
                output.update({'point_coords': pc, 'point_labels': pl, 'boxes': box})
                
        return output
