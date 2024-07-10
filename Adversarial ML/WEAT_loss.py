# Some code implementation is based from https://github.com/yaodongyu/TRADES
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torchvision

def energy_x(logits):
  return -torch.logsumexp(logits, dim=1)


def compute_weights(energy):
  return (1/(torch.log(1+torch.exp(torch.abs(energy)))))


def WEAT_adv(model,
              x_natural,
              y,
              optimizer,
              step_size=0.007,
              epsilon=0.031,
              perturb_steps=10,
              beta=7.0):
    """
    Compute the energy-weighted loss for adversarial training. i.e WEAT_adv in paper

    Parameters:
    -----------
    model : torch.nn.Module
        The neural network model to be trained.
    x_natural : torch.Tensor
        The natural (non-adversarial) input data.
    y : torch.Tensor
        The ground truth labels for the input data.
    optimizer : torch.optim.Optimizer
        The optimizer used to update the model parameters.
    step_size : float
        The step size for generating adversarial examples.
    epsilon : float
        The maximum perturbation allowed for the adversarial examples.
    perturb_steps : int
        The number of steps to use for generating adversarial examples.
    beta : float
        The weight for the KL-divergence loss component.
    Returns:
    --------
    loss : torch.Tensor
        The total computed weighted loss combining adversarial and robust loss components.
    Notes:
    ------
    This function is designed for adversarial training with energy-based weighting.
    It generates adversarial examples using PGD witk KL-Divergence(TRADES), and returns the loss.
    """
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    kl_sample = nn.KLDivLoss(reduction='none')

    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    x_natural=x_natural.clone().detach().cuda()
    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl = criterion_kl(torch.log(F.softmax(model(x_adv), dim=1) + 1e-12),
                                   F.softmax(model(x_natural), dim=1))
        grad = torch.autograd.grad(loss_kl, x_adv, retain_graph=False, create_graph=False)[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    
   
    model.train()
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

    # zero gradient
    optimizer.zero_grad()

    # calculate robust loss
    logits = model(x_natural)
    adv_logits = model(x_adv)
    e_x=  energy_x(logits).detach()
    weights = compute_weights(e_x)    

    loss_adv =  (1.0 / batch_size) * torch.sum(F.cross_entropy(adv_logits, y, reduction='none') *  weights)
    loss_kl = (1.0 / batch_size) * torch.sum(
    torch.sum(kl_sample(torch.log(F.softmax(adv_logits, dim=1) + 1e-12), F.softmax(logits, dim=1)),dim=1) * weights)
  
    loss = loss_adv + (beta * loss_kl) 
    return loss


def WEAT_nat(model,
              x_natural,
              y,
              optimizer,
              step_size=0.007,
              epsilon=0.031,
              perturb_steps=10,
              beta=6.0):
    """
    Compute the energy-weighted loss for adversarial training i.e WEAT_nat in paper
    """
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    kl_sample = nn.KLDivLoss(reduction='none')

    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    x_natural=x_natural.clone().detach().cuda()
    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl = criterion_kl(torch.log(F.softmax(model(x_adv), dim=1) + 1e-12),
                                   F.softmax(model(x_natural), dim=1))
        grad = torch.autograd.grad(loss_kl, x_adv, retain_graph=False, create_graph=False)[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    
   
    model.train()
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

    # zero gradient
    optimizer.zero_grad()

    # calculate robust loss
    logits = model(x_natural)
    adv_logits = model(x_adv)
    e_x=  energy_x(logits).detach()
    weights = compute_weights(e_x)    

    loss_ce =  (1.0 / batch_size) * torch.sum(F.cross_entropy(logits, y, reduction='none') *  weights)
    loss_kl = (1.0 / batch_size) * torch.sum(
    torch.sum(kl_sample(torch.log(F.softmax(adv_logits, dim=1) + 1e-12), F.softmax(logits, dim=1)),dim=1) * weights)
  
    loss = loss_ce + (beta * loss_kl) 
    return loss
