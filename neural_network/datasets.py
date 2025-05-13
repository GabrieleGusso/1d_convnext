# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 2022 Zhuang Liu (Meta) - liuzhuangthu@gmail.com

import os
import dataloader
from torchvision import datasets, transforms
import torch
from torchvision.transforms import Lambda

from timm.data.constants import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    IMAGENET_INCEPTION_MEAN,
    IMAGENET_INCEPTION_STD,
)
from timm.data import create_transform


def build_dataset(is_train, args):
    if args.data_set == "CIFAR":
        transform = build_transform(is_train, args)
        dataset = datasets.CIFAR100(
            args.data_path, train=is_train, transform=transform, download=True
        )
        nb_classes = 100
    elif args.data_set == "IMNET":
        transform = build_transform(is_train, args)
        print("reading from datapath", args.data_path)
        root = os.path.join(args.data_path, "train" if is_train else "val")
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == "image_folder":
        transform = build_transform(is_train, args)
        root = args.data_path if is_train else args.eval_data_path
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = args.nb_classes
        assert len(dataset.class_to_idx) == nb_classes
    elif "gauss" in args.data_set or "real" in args.data_set:
        root = os.path.join(args.data_path, "train" if is_train else "val")
        transform = Lambda(lambda x: torch.tensor(x))
        target_transform = Lambda(lambda y: torch.tensor(y))
        print("Transform = Lambda(lambda x: torch.tensor(x))")
        print("Target transform = Lambda(lambda y: torch.tensor(y))")
        dataset = dataloader.MatVectorFolder(
            root, transform=transform, target_transform=target_transform
        )
        nb_classes = args.nb_classes
        assert len(dataset.class_to_idx) == nb_classes
    else:
        raise NotImplementedError()

    print("Number of the class = %d" % nb_classes)
    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = (
        IMAGENET_INCEPTION_MEAN
        if not imagenet_default_mean_and_std
        else IMAGENET_DEFAULT_MEAN
    )
    std = (
        IMAGENET_INCEPTION_STD
        if not imagenet_default_mean_and_std
        else IMAGENET_DEFAULT_STD
    )

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        # warping (no cropping) when evaluated at 384 or larger
        if args.input_size >= 384:
            t.append(
                transforms.Resize(
                    (args.input_size, args.input_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
            )
            print(f"Warping {args.input_size} size input images...")
        else:
            if args.crop_pct is None:
                args.crop_pct = 224 / 256
            size = int(args.input_size / args.crop_pct)
            t.append(
                # to maintain same ratio w.r.t. 224 images
                transforms.Resize(
                    size, interpolation=transforms.InterpolationMode.BICUBIC
                ),
            )
            t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")
    return transforms.Compose(t)
