import os
import random

import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode


def _resolve_train_list(data_volume, default_train_list):
    if data_volume == "1":
        return "train_1.txt"
    if data_volume == "10":
        return "train_10.txt"
    if data_volume == "100":
        return default_train_list
    return default_train_list


def _is_path_token(token):
    if token.isdigit():
        return False
    return ("/" in token) or ("\\" in token) or ("." in token)


class _ENDOBase(data.Dataset):
    def __init__(
        self,
        root,
        data_volume,
        split,
        train_list,
        val_list,
        test_list,
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.list_image_paths = []
        self.list_image_labels = []
        self.list_mask_paths = []

        train_label_data = _resolve_train_list(data_volume, train_list)
        if split == "train":
            list_file = train_label_data
        elif split == "val":
            list_file = val_list
        elif split == "test":
            list_file = test_list
        else:
            raise ValueError("Unknown split: {}".format(split))

        with open(os.path.join(root, list_file), "r") as file_descriptor:
            for line in file_descriptor:
                line_items = line.split()
                if len(line_items) < 2:
                    continue
                image_path = line_items[0]
                if len(line_items) >= 3 and _is_path_token(line_items[-1]):
                    label_items = line_items[1:-1]
                    mask_path = line_items[-1]
                else:
                    label_items = line_items[1:]
                    mask_path = None
                image_label = [int(item) for item in label_items]

                if not os.path.isabs(image_path):
                    image_path = os.path.join(root, image_path)
                if mask_path is not None and not os.path.isabs(mask_path):
                    mask_path = os.path.join(root, mask_path)

                self.list_image_paths.append(image_path)
                self.list_image_labels.append(image_label)
                self.list_mask_paths.append(mask_path)

    def __len__(self):
        return len(self.list_image_paths)


class ENDO(_ENDOBase):
    def __init__(
        self,
        root,
        data_volume,
        split="train",
        transform=None,
        train_list="train.txt",
        val_list="val.txt",
        test_list="test.txt",
    ):
        super().__init__(
            root=root,
            data_volume=data_volume,
            split=split,
            train_list=train_list,
            val_list=val_list,
            test_list=test_list,
        )
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.list_image_paths[index]
        image_data = Image.open(image_path).convert("RGB")
        image_label = torch.FloatTensor(self.list_image_labels[index])

        if self.transform is not None:
            image_data = self.transform(image_data)

        return image_data, image_label


class ENDO_R(_ENDOBase):
    def __init__(
        self,
        root,
        data_volume,
        split="train",
        img_size=224,
        train_list="train-r.txt",
        val_list="val.txt",
        test_list="test.txt",
    ):
        super().__init__(
            root=root,
            data_volume=data_volume,
            split=split,
            train_list=train_list,
            val_list=val_list,
            test_list=test_list,
        )
        self.img_size = img_size
        self.color_jitter = transforms.ColorJitter(brightness=0.15, contrast=0.15)
        self.to_grayscale = transforms.Grayscale(num_output_channels=3)
        self.normalize = transforms.Normalize(mean=[0.4978], std=[0.2449])

    def _load_mask(self, mask_path, size):
        if mask_path is None or not os.path.exists(mask_path):
            return Image.new("L", size, 0)
        return Image.open(mask_path).convert("L")

    def _transform_train(self, image, mask):
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            image, scale=(0.8, 1.0), ratio=(0.9, 1.1)
        )
        image = TF.resized_crop(
            image, i, j, h, w, (self.img_size, self.img_size), interpolation=InterpolationMode.BILINEAR
        )
        mask = TF.resized_crop(
            mask, i, j, h, w, (self.img_size, self.img_size), interpolation=InterpolationMode.NEAREST
        )

        angle = random.uniform(-10, 10)
        image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
        mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST, fill=0)

        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        image = self.color_jitter(image)
        image = self.to_grayscale(image)
        image = TF.to_tensor(image)
        image = self.normalize(image)

        mask = TF.to_tensor(mask)
        mask = (mask > 0.5).float()
        return image, mask

    def _transform_eval(self, image, mask):
        image = TF.resize(image, self.img_size, interpolation=InterpolationMode.BILINEAR)
        image = TF.center_crop(image, (self.img_size, self.img_size))
        mask = TF.resize(mask, self.img_size, interpolation=InterpolationMode.NEAREST)
        mask = TF.center_crop(mask, (self.img_size, self.img_size))

        image = self.to_grayscale(image)
        image = TF.to_tensor(image)
        image = self.normalize(image)

        mask = TF.to_tensor(mask)
        mask = (mask > 0.5).float()
        return image, mask

    def __getitem__(self, index):
        image_path = self.list_image_paths[index]
        mask_path = self.list_mask_paths[index]

        image = Image.open(image_path).convert("RGB")
        mask = self._load_mask(mask_path, image.size)
        image_label = torch.FloatTensor(self.list_image_labels[index])

        if self.split == "train":
            image, mask = self._transform_train(image, mask)
        else:
            image, mask = self._transform_eval(image, mask)

        return image, image_label, mask
