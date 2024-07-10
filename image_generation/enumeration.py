from enum import Enum


class BenchmarkDataset(Enum):
    cifar_10 = "cifar10"
    cifar_100 = "cifar100"
    imagenet = "imagenet"


dataset_image_size = {"cifar10": 32, "cifar100": 32, "svhn": 28}

dataset_num_classes = {"cifar10": 10, "cifar100": 100, "svhn": 10}
