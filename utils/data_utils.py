import logging

import torch
from torchvision import transforms
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler

from utils.dist_util import get_world_size
from .my_dataset import ENDO, ENDO_R

logger = logging.getLogger(__name__)


def _build_base_transforms(args):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomRotation(degrees=10),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4978], std=[0.2449]),
    ])
    transform_eval = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.CenterCrop((args.img_size, args.img_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4978], std=[0.2449]),
    ])
    return transform_train, transform_eval


def _make_loader(dataset, sampler, batch_size):
    return DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size // get_world_size(),
        num_workers=10,
        pin_memory=True,
    ) if dataset is not None else None


def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    transform_train, transform_eval = _build_base_transforms(args)

    if args.stage in ["test", "val"]:
        split = "test" if args.stage == "test" else "val"
        dataset = ENDO(
            root=args.dataset_path,
            data_volume=args.data_volume,
            split=split,
            transform=transform_eval,
            train_list=args.train_list,
            val_list=args.val_list,
            test_list=args.test_list,
        )
        print("testset", len(dataset))
        if args.local_rank == 0:
            torch.distributed.barrier()
        return _make_loader(dataset, SequentialSampler(dataset), args.eval_batch_size)

    trainset = ENDO(
        root=args.dataset_path,
        data_volume=args.data_volume,
        split="train",
        transform=transform_train,
        train_list=args.train_list,
        val_list=args.val_list,
        test_list=args.test_list,
    )
    valset = ENDO(
        root=args.dataset_path,
        data_volume=args.data_volume,
        split="val",
        transform=transform_eval,
        train_list=args.train_list,
        val_list=args.val_list,
        test_list=args.test_list,
    )
    print("train_loader", len(trainset))
    print("test_loader", len(valset))
    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    val_sampler = SequentialSampler(valset)
    train_loader = _make_loader(trainset, train_sampler, args.train_batch_size)
    val_loader = _make_loader(valset, val_sampler, args.eval_batch_size)
    return train_loader, val_loader


def get_loader_r1(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    if args.stage in ["test", "val"]:
        split = "test" if args.stage == "test" else "val"
        dataset = ENDO_R(
            root=args.dataset_path,
            data_volume=args.data_volume,
            split=split,
            img_size=args.img_size,
            train_list=args.train_list,
            val_list=args.val_list,
            test_list=args.test_list,
        )
        print("testset", len(dataset))
        if args.local_rank == 0:
            torch.distributed.barrier()
        return _make_loader(dataset, SequentialSampler(dataset), args.eval_batch_size)

    trainset = ENDO_R(
        root=args.dataset_path,
        data_volume=args.data_volume,
        split="train",
        img_size=args.img_size,
        train_list=args.train_list,
        val_list=args.val_list,
        test_list=args.test_list,
    )
    valset = ENDO_R(
        root=args.dataset_path,
        data_volume=args.data_volume,
        split="val",
        img_size=args.img_size,
        train_list=args.train_list,
        val_list=args.val_list,
        test_list=args.test_list,
    )
    print("train_loader", len(trainset))
    print("test_loader", len(valset))
    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    val_sampler = SequentialSampler(valset)
    train_loader = _make_loader(trainset, train_sampler, args.train_batch_size)
    val_loader = _make_loader(valset, val_sampler, args.eval_batch_size)
    return train_loader, val_loader
