import os
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Set, Tuple, Union

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch import nn


PREPROCESSINGS = {
    "Res256Crop224": transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
    ),
    "Crop288": transforms.Compose([transforms.CenterCrop(288), transforms.ToTensor()]),
    None: transforms.Compose([transforms.ToTensor()]),
    "Res224": transforms.Compose([transforms.Resize(224), transforms.ToTensor()]),
    "BicubicRes256Crop224": transforms.Compose(
        [
            transforms.Resize(
                256, interpolation=transforms.InterpolationMode("bicubic")
            ),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    ),
}


def _load_dataset(
    dataset: Dataset, n_examples: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = 100
    test_loader = data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    x_test, y_test = [], []
    for i, (x, y) in enumerate(test_loader):
        x_test.append(x)
        y_test.append(y)
        if n_examples is not None and batch_size * i >= n_examples:
            break
    x_test_tensor = torch.cat(x_test)
    y_test_tensor = torch.cat(y_test)

    if n_examples is not None:
        x_test_tensor = x_test_tensor[:n_examples]
        y_test_tensor = y_test_tensor[:n_examples]

    return x_test_tensor, y_test_tensor


def load_svhn(
    n_examples: Optional[int] = 5000,
    data_dir: str = "./data",
    transforms_test: Callable = PREPROCESSINGS["Res256Crop224"],
) -> Tuple[torch.Tensor, torch.Tensor]:

    import torchvision.datasets

    dataset = torchvision.datasets.SVHN(
        root=data_dir, split="test", transform=transforms_test, download=True
    )
    len_dataset = len(dataset)
    print(f"len_dataset: {len_dataset}")
    if n_examples < len_dataset:
        n_examples = len_dataset
    return _load_dataset(dataset, n_examples)


def load_tiny_imagenet(
    n_examples: Optional[int] = 10000,
    data_dir: str = "/data1/omnai/CV/TINY_IMAGENET/tiny-imagenet-200/",
    transforms_test: Callable = PREPROCESSINGS["Res256Crop224"],
) -> Tuple[torch.Tensor, torch.Tensor]:
    from robustbenchmaster.robustbench.load_tiny_imagenet import load_tinyimagenet

    train_dataset, test_dataset = load_tinyimagenet(data_dir=data_dir)
    return _load_dataset(test_dataset, n_examples)


