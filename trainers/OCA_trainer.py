#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#

from math import sqrt
from typing import Dict, Tuple, Callable

import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.logging_utils import AverageMeter
from utils.eval_utils import accuracy

cosine_criterion = nn.CosineSimilarity(dim=1)
entropy_criterion = nn.CrossEntropyLoss()
cosine_loss = torch.nn.CosineEmbeddingLoss()

class OCATrainer():
    """Class to train and evaluate regularized new model 
    with a given old model."""

    def train(self,
              train_loader: torch.utils.data.DataLoader,
              model: nn.Module,
              criterion: Callable,
              optimizer: torch.optim.Optimizer,
              device: torch.device,
              accelerator,
              pseudo_classifier=None) -> Tuple[float, float, float]:
        """Run one epoch of training.

        :param train_loader: Data loader to train the model.
        :param model: Model to be trained.
        :param old_feature_dict: Dictionary of old feature, 
            with the keys being the paths of images.
        ::param new_feature_dict: Dictionary of new feature,
            with the keys being the paths of images.
        :param criterion: Loss criterion module.
        :param optimizer: A torch optimizer object.
        :param device: Device the model is on.
        :param accelerator: Huggingface accelerator
        :param pseudo_classifier: A pseudo-classifier for BCT 
            influence loss
        :param lambda_1: a multiplier on the cosine matching 
            loss with new_feature
        :param lambda_2: a multiplier on the BCT influence loss
        :param lambda_3: a multiplier on the cosine matching
            loss with old_feature
        :return: average loss on current epoch.
        """
        losses = AverageMeter("Loss", ":.3f")
        losses_bct = AverageMeter("Loss_BCT", ":.3f")
        losses_ce = AverageMeter("Loss_CE", ":.3f")
        losses_cosine = AverageMeter("Loss_cosine", ":.3f")

        model = model.train().to(device)
        # new_model = new_model.eval().to(device)

        for i, (paths, (images, target)) in tqdm.tqdm(
                enumerate(train_loader), ascii=True, total=len(train_loader)
        ):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            outputs = model(images)
            #output
            output = outputs[0]
            #bct_feature
            old_feature = outputs[1]

            old_feature = old_feature.view(old_feature.size(0), -1)
            old_feature_norm = F.normalize(old_feature, dim=1)

            loss_cosine = cosine_loss(pseudo_classifier[target], old_feature_norm, torch.ones(old_feature_norm.size(0)).to(device))

            # #normale loss CE
            loss_ce = criterion(output, target)

            #BCT influence loss
            if pseudo_classifier is not None:
                pseudo_output = old_feature_norm @ pseudo_classifier.transpose(0, 1)
                loss_bct = criterion(pseudo_output, target)

            loss = 5 * loss_ce + 10* loss_bct +  5 * loss_cosine

            losses.update(loss.item(), images.size(0))
            losses_bct.update(loss_bct.item(), images.size(0))
            losses_ce.update(loss_ce.item(), images.size(0))
            losses_cosine.update(loss_cosine.item(), images.size(0))

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

        return losses.avg, losses_bct.avg, losses_ce.avg, losses_cosine.avg

    def validate(self,
                 val_loader: torch.utils.data.DataLoader,
                 model: nn.Module,
                 device: torch.device) -> Tuple[float, float, float]:
        """Run validation.

        :param val_loader: Data loader to evaluate the model.
        :param model: Model to be evaluated.
        :param old_feature_dict: Dictionary of old feature, 
            with the keys being the paths of images.
        ::param new_feature_dict: Dictionary of new feature,
            with the keys being the paths of images.
        :param criterion: Loss criterion module.
        :param device: Device the model is on.
        :return: average of loss and top-1 on current epoch.
        """
        losses = AverageMeter("Loss", ":.3f")
        top1 = AverageMeter("Acc@1", ":6.2f")

        model = model.eval()

        with torch.no_grad():
            for i, (paths, (images, target)) in tqdm.tqdm(
                    enumerate(val_loader), ascii=True, total=len(val_loader)
            ):
                images = images.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                model = model.to(device)

                outputs = model(images)
                output = outputs[0]
                acc1, acc5 = accuracy(output, target, topk=(1, 5))

                top1.update(acc1.item(), images.size(0))

        return losses.avg, top1.avg


def build_feature_dict(loader: torch.utils.data.DataLoader,
                       model: nn.Module,
                       device: torch.device,
                       feature_dict) -> Tuple[float, float, float]:
    """
    return a dictionary of saved features
    """
    model = model.eval().to(device)

    for i, (paths, (images, target)) in tqdm.tqdm(
            enumerate(loader), ascii=True, total=len(loader)
    ):
        images = images.to(device, non_blocking=True)
        model = model

        try:
            images = model.feature_extractor(
                images.cpu()).to(device, non_blocking=True)
        except AttributeError:
            pass

        with torch.no_grad():
            features = model(images)[1].cpu()
        features = features.reshape(features.size(0), -1)
        features = F.normalize(features, dim=1)
        for path, feature in zip(paths, features):
            feature_dict[path] = feature

    return feature_dict