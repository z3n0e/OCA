from typing import List
from torchvision.datasets import CIFAR100, CIFAR10
import random
import numpy as np
import torch
from torchvision import datasets
from torchvision import transforms


class SubImageCIFAR100:
    """Class to support training on subset of classes."""

    def __init__(self, name: str, data_root: str, num_workers: int,
                 batch_size: int,
                 num_classes=None) -> None:
        
        super(SubImageCIFAR100, self).__init__()

        normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )

        transform_train = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                normalize,
            ]
        )

        self.train_dataset = CIFAR100Subset(num_classes=num_classes, root="./data_store",
                                            train=True,
                                            download=True,
                                            transform=transform_train,)
        
        self.val_dataset = CIFAR100Subset(root="./data_store",
                                            train=False,
                                            download=True,
                                            transform=transform_test,)

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
        )

        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
        )


class CIFAR100Subset(CIFAR100):
    def __init__(self, num_classes=None, **kwargs):
        super().__init__(**kwargs)
        if num_classes is None:
            self.subset = len(self.classes)
        else:
            self.subset = num_classes
        print(len(self.classes))
        print(self.subset)
        assert self.subset <= len(self.classes)
        assert self.subset >= 0

        self.aligned_indices = []
        for idx, label in enumerate(self.targets):
            if label < self.subset:
                self.aligned_indices.append(idx)

    def get_class_names(self):
        return [self.classes[i] for i in range(self.subset)]

    def __len__(self):
        return len(self.aligned_indices)

    def __getitem__(self, item):
        return str(self.aligned_indices[item]), super().__getitem__(self.aligned_indices[item])