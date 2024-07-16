import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
import wandb
import numpy as np
from WEAT_loss import WEAT_nat, WEAT_adv



def adv_train(args, model, device, train_loader, optimizer,scheduler, epoch):
    model.train()
    total_loss= 0
  
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        if args.adv_loss=='WEAT_nat':
            loss  = WEAT_nat(model=model,
                                      x_natural=data,
                                      y=target,
                                      optimizer=optimizer,
                                      step_size=args.step_size,
                                      epsilon=args.epsilon,
                                      perturb_steps=args.num_steps,
                                      beta=args.beta )
        elif args.adv_loss=='WEAT_adv':
            loss = WEAT_adv(model=model,
                                      x_natural=data,
                                      y=target,
                                      optimizer=optimizer,
                                      step_size=args.step_size,
                                      epsilon=args.epsilon,
                                      perturb_steps=args.num_steps,
                                      beta=args.beta)
        else:
            print("Adversarial Loss not Valid")
        
        loss.backward()
        optimizer.step()
        scheduler.step()
    
        total_loss = total_loss + loss.item()
        
    total_loss/= len(train_loader)
    return total_loss


def eval_std(model, device, loader):
    model.eval()
    eval_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            eval_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    eval_loss /= len(loader.dataset)
    return eval_loss
