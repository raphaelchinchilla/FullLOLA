from deepillusion.torchattacks._utils import clip
import torch.nn.functional as F
import torch.nn as nn
import torch
import pdb
import numpy as np
from tqdm import tqdm
"""
Description: Training and testing functions for neural models

functions:
    train: Performs a single training epoch (if attack_args is present adversarial training)
    test: Evaluates model by computing accuracy (if attack_args is present adversarial testing)
"""


__all__ = ['adversarial_epoch', 'adversarial_test']



def adversarial_epoch(model, train_loader, optimizer, scheduler=None, adversarial_args=None):
    """
    Description: Single epoch,
        if adversarial args are present then adversarial training.
    Input :
        model : Neural Network               (torch.nn.Module)
        train_loader : Data loader           (torch.utils.data.DataLoader)
        optimizer : Optimizer                (torch.nn.optimizer)
        scheduler: Scheduler (Optional)      (torch.optim.lr_scheduler.CyclicLR)
        adversarial_args :
            attack:                          (deepillusion.torchattacks)
            attack_args:
                attack arguments for given attack except "x" and "y_true"
    Output:
        train_loss : Train loss              (float)
        train_accuracy : Train accuracy      (float)
    """

    model.train()

    device = model.parameters().__next__().device

    eps = adversarial_args["attack_args"]["attack_params"]["eps"]

    beta = adversarial_args["attack_args"]["attack_params"]["step_size"]  # attack learning rate


    cross_ent = nn.CrossEntropyLoss()



    train_loss = 0
    train_correct = 0
    for data, target in train_loader:

        data, target = data.to(device), target.to(device)

        # Adversary  perturbation
        if adversarial_args["attack_args"]["attack_params"]["random_start"]:
            e = clip(2*eps*(torch.rand_like(data)-0.5), adversarial_args["attack_args"]["data_params"]
                     ["x_min"]-data, adversarial_args["attack_args"]["data_params"]["x_max"]-data).requires_grad_(True)
        else:
            e = torch.zeros_like(data, requires_grad=True)

        for iteration in range(0, adversarial_args["attack_args"]["attack_params"]["num_steps"]):

            if device.type == "cuda":
                y_hat = model(data + e).type(torch.cuda.DoubleTensor)
            else:
                y_hat = model(data + e)

            # Computing the prediction of the adversarial attack i.e., d_y(x,y)
            y_hat = y_hat.view(-1, y_hat.shape[-1])
            loss_max = cross_ent(y_hat, target)
            grad_max = torch.autograd.grad(loss_max, e, create_graph=True)[0] # create_graph means that grad_max can be differentiate

            # Computing the descent direction for the FullLOLA cost i.e. f(x,y+ d_y(x,y))
            # The perturbed data is clamped such that the perturbation belongs to (-eps,eps) and such that x+e belongs to (x_min,x_max)
            data_perturbed = (data+(e+beta*grad_max).clamp(-eps, eps)).clamp(
                adversarial_args["attack_args"]["data_params"]["x_min"], adversarial_args["attack_args"]["data_params"]["x_max"])
            optimizer.zero_grad()
            output = model(data_perturbed)
            loss = cross_ent(output, target)
            loss.backward(retain_graph=True)
            optimizer.step()

            # avoiding computation of a last useless step
            if iteration < adversarial_args["attack_args"]["attack_params"]["num_steps"]-1:

                if device.type == "cuda":
                    y_hat = model(data + e).type(torch.cuda.DoubleTensor)
                else:
                    y_hat = model(data + e)

                # Computing the adverserial attack for the updated model (i.e., weights and biases)
                y_hat = y_hat.view(-1, y_hat.shape[-1])
                loss_max = cross_ent(y_hat, target)
                grad_max = torch.autograd.grad(loss_max, e)[0]

                with torch.no_grad():
                    e += beta*grad_max
                    e = e.clamp_(-eps, eps)
                    e.data = clip(e, adversarial_args["attack_args"]["data_params"]["x_min"] -
                                  data, adversarial_args["attack_args"]["data_params"]["x_max"] - data)

        if scheduler:
            scheduler.step()

        train_loss += loss.item() * data.size(0)
        pred_adv = output.argmax(dim=1, keepdim=True)
        train_correct += pred_adv.eq(target.view_as(pred_adv)).sum().item()

    train_size = len(train_loader.dataset)

    return train_loss/train_size, train_correct/train_size


def adversarial_test(model, test_loader, adversarial_args=None, verbose=False, progress_bar=False):
    """
    Description: Evaluate model with test dataset,
        if adversarial args are present then adversarially perturbed test set.
    Input :
        model : Neural Network               (torch.nn.Module)
        test_loader : Data loader            (torch.utils.data.DataLoader)
        adversarial_args :                   (dict)
            attack:                          (deepillusion.torchattacks)
            attack_args:                     (dict)
                attack arguments for given attack except "x" and "y_true"
        verbose: Verbosity                   (Bool)
        progress_bar: Progress bar           (Bool)
    Output:
        train_loss : Train loss              (float)
        train_accuracy : Train accuracy      (float)
    """

    device = model.parameters().__next__().device

    model.eval()

    test_loss = 0
    test_correct = 0
    if progress_bar:
        iter_test_loader = tqdm(
            iterable=test_loader,
            unit="batch",
            leave=False)
    else:
        iter_test_loader = test_loader

    for data, target in iter_test_loader:

        data, target = data.to(device), target.to(device)

        # Adversary
        if adversarial_args and adversarial_args["attack"]:
            adversarial_args["attack_args"]["net"] = model
            adversarial_args["attack_args"]["x"] = data
            adversarial_args["attack_args"]["y_true"] = target
            perturbs = adversarial_args['attack'](**adversarial_args["attack_args"])
            data += perturbs

        output = model(data)

        cross_ent = nn.CrossEntropyLoss()
        test_loss += cross_ent(output, target).item() * data.size(0)

        pred = output.argmax(dim=1, keepdim=True)
        test_correct += pred.eq(target.view_as(pred)).sum().item()

    test_size = len(test_loader.dataset)

    return test_loss/test_size, test_correct/test_size
