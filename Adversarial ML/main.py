from __future__ import print_function
import logging
import time
import os
import argparse
from parser import get_parser
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import wandb
from wandb.util import generate_id
from models.resnet import *
from models.ti_resnet import *
from pgd_attack import eval_adv_test_whitebox
from utils import adv_train, eval_std
from load_tiny_imagenet import load_tinyimagenet
import warnings
warnings.filterwarnings("ignore")


# Initialize and configure WandB
def initialize_wandb():
    random_run_name = generate_id()
    print(random_run_name)
    run = wandb.init(project='<enter_wandb_project>', name=random_run_name)
    return run, random_run_name

# Disable non-deterministic operations for better reproducibility
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Define a custom dataset class that is used to load synthesized data.
class Generated_Dataset(torch.utils.data.Dataset):
    def __init__(self, file_path):
        data = np.load(file_path)
        self.images = data['image']
        self.labels = data['label']
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = np.array(self.images[index]).astype(np.uint8)
        image = self.transform(image)
        label = np.array(self.labels[index])
        return image, label

# Load datasets based on the chosen dataset type
def load_datasets(args, transform_train):
    if args.dataset == 'CIFAR-10':
        generated_file_path = '<path_to_generated_data>'
        num_samples = 50000
        path = '<path_to_data>'
        num_classes = 10
        trainset = torchvision.datasets.CIFAR10(root=path, train=True, download=False, transform=transform_train)
    elif args.dataset == 'CIFAR-100':
        generated_file_path = '<path_to_generated_data>'
        num_samples = 50000
        path ='<path_to_data>'
        num_classes = 100
        trainset = torchvision.datasets.CIFAR100(root=path, train=True, download=False, transform=transform_train)
    elif args.dataset == 'SVHN':
        generated_file_path = '<path_to_generated_data>'
        num_samples = 73257
        path = '<path_to_data>'
        num_classes = 10
        trainset = torchvision.datasets.SVHN(root=path, split='train', download=False, transform=transform_train)
    elif args.dataset == 'tiny_imagenet':
        generated_file_path = '<path_to_generated_data>'
        num_samples = 100000
        path = '<path_to_data>'
        num_classes = 200
        trainset, _ = load_tinyimagenet(path, use_augmentation='base')
    else:
        raise ValueError("Invalid dataset mentioned")
    
    generated_dataset = Generated_Dataset(generated_file_path)
    return trainset, generated_dataset, num_classes,path

# Initialize model based on architecture
def initialize_model(args, device, num_classes):
    if args.architecture == 'PreActResNet18':
        model = PreActResNet18(classes=num_classes).to(device)
    else:
        if args.dataset == 'tiny_imagenet':
            model = ti_ResNet18(classes=num_classes).to(device)
        else:
            model = ResNet18(classes=num_classes).to(device)
    return model

# Initialize optimizer and scheduler
def initialize_optimizer_and_scheduler(args, model, num_samples):
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    update_steps = int(np.floor(num_samples / args.batch_size) + 1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, pct_start=0.25,
                                                    steps_per_epoch=update_steps, epochs=int(args.epochs))
    return optimizer, scheduler

# Evaluation function
def evaluate(model, device, val_loader, epoch, args):
    print('Evaluating with PGD...')
    val_natural_err_total, val_robust_err_total = eval_adv_test_whitebox(model, device, val_loader, args)
    return val_natural_err_total, val_robust_err_total

# Evaluate standard performance on original data
def evaluate_standard(model, device, train_loader, val_loader):
    print('Evaluating on original data...')
    train_loss = eval_std(model, device, train_loader)
    val_loss = eval_std(model, device, val_loader)
    return train_loss, val_loss

# display results  and log results to WandB
def show_and_log_results(run,epoch, loss, val_natural_err_total, val_robust_err_total,
                         train_loss, val_loss, last_lr):
    run.log({'epoch': epoch, 'loss': loss,
             'Validation_Error': val_natural_err_total, 'Validation_Error_Adv': val_robust_err_total,
             'Training_Loss_Org': train_loss, 'Val_Loss_Org': val_loss, 'learning_rate': last_lr})

    print(f"{'='*60}\n"
          f"{'Training Metrics':^60}\n"
          f"{'-'*60}\n"
          f"Epoch: {epoch}, Total Train Loss: {loss:.4f}\n"
          f"{'Validation Metrics':^60}\n"
          f"{'-'*60}\n"
          f"Val Natural Error: {val_natural_err_total:.4f}, Val Robust Error: {val_robust_err_total:.4f}\n"
          f"{'Standard Evaluation Metrics':^60}\n"
          f"{'-'*60}\n"
          f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}\n"
          f"Learning Rate: {last_lr:.6f}\n"
          f"{'='*60}",flush=True)


# Save the best model based on the least test robust error and return the model instance
def save_best_model(model, epoch, val_robust_err_total, best_robust_err, current_run_name):
    if val_robust_err_total < best_robust_err:
        best_robust_err = val_robust_err_total
        save_path = os.path.join('<enter_path_to_save_model>', current_run_name)
        os.makedirs(save_path, exist_ok=True)
        model_save_path = os.path.join(save_path, f'best_model_epoch_{epoch}.pth')
        torch.save(model.state_dict(), model_save_path)
        print(f"New best model saved with robust error {val_robust_err_total:.4f} at epoch {epoch}")
        return model, best_robust_err, epoch
    return None, best_robust_err,None



# Main function
def main():
    run, random_run_name = initialize_wandb()
    parser = get_parser()
    args = parser.parse_args()
    wandb.config.update(vars(args))
    set_random_seed(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(0.5), transforms.ToTensor()])
    trainset, generated_dataset, num_classes,path = load_datasets(args, transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(generated_dataset, batch_size=args.test_batch_size, shuffle=False)

    model = initialize_model(args, device, num_classes)
    optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(trainset))

    best_robust_err = float('inf')
    best_model = None
    best_epoch = None

    t0=time.time()

    for epoch in range(1, args.epochs + 1):
        print(f'{"="*60}\n{"Epoch Started":^60}\n{"="*60}')
        last_lr =  scheduler.get_last_lr()[0]

        loss = adv_train(args, model, device, train_loader, optimizer, scheduler, epoch)

        val_natural_err_total, val_robust_err_total = evaluate(model, device, val_loader, epoch, args)
        train_loss, val_loss = evaluate_standard(model, device, train_loader, val_loader)


        show_and_log_results(run, epoch, loss, val_natural_err_total, val_robust_err_total, train_loss, val_loss, last_lr)
        if epoch > args.save_model:
            model_instance, best_robust_err, saved_epoch = save_best_model(model, epoch, val_robust_err_total, best_robust_err, random_run_name)
            if model_instance:
                best_model = model_instance
            if saved_epoch:
                best_epoch = saved_epoch
    wandb.finish()


if __name__ == '__main__':
    main()
