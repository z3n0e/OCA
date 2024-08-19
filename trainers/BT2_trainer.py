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

class BT2Trainer():
    """Class to train and evaluate regularized new model 
    with a given old model."""

    def train(self,
              train_loader: torch.utils.data.DataLoader,
              model: nn.Module,
              old_feature_dict: Dict,
              new_feature_dict: Dict,
              criterion: Callable,
              optimizer: torch.optim.Optimizer,
              device: torch.device,
              accelerator,
              pseudo_classifier=None,
              lambda_1=1,
              lambda_2=1,
              lambda_3=1) -> Tuple[float, float, float]:
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

        model = model.train().to(device)
        # new_model = new_model.eval().to(device)

        for i, (paths, (images, target)) in tqdm.tqdm(
                enumerate(train_loader), ascii=True, total=len(train_loader)
        ):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            outputs = model(images)
            old_feature = outputs[1]
            feature = outputs[2]
            output = outputs[0]
            feature = feature.view(feature.size(0), -1)
            old_feature = old_feature.view(feature.size(0), -1)
            old_feature = F.normalize(old_feature, dim=1)
            feature = F.normalize(feature, dim=1)

            phi_old = []
            phi_p = []
            for path in paths:
                phi_old.append(old_feature_dict[path].view(1, -1))
                phi_p.append(new_feature_dict[path].view(1, -1))
            phi_old = torch.cat(phi_old, dim=0).to(device)
            phi_p = torch.cat(phi_p, dim=0).to(device)

            cosine_loss = lambda_3 * \
                (1 - torch.mean(torch.sum(phi_old * old_feature, dim=1)))
            cosine_loss += lambda_1 * \
                (1 - torch.mean(torch.sum(phi_p * feature, dim=1)))

            loss = 1*entropy_criterion(output, target)
            if pseudo_classifier is not None:
                pseudo_output = old_feature.view(old_feature.size(
                    0), -1) @ pseudo_classifier.transpose(0, 1)
                loss += lambda_2 * entropy_criterion(pseudo_output, target)
            loss += cosine_loss

            losses.update(loss.item(), images.size(0))

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

        return losses.avg

    def validate(self,
                 val_loader: torch.utils.data.DataLoader,
                 model: nn.Module,
                 old_feature_dict: Dict,
                 new_feature_dict: Dict,
                 criterion: Callable,
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
                old_feature = outputs[1]
                feature = outputs[2]
                output = outputs[0]
                feature = feature.view(feature.size(0), -1)
                old_feature = old_feature.view(feature.size(0), -1)
                feature = F.normalize(feature, dim=1)
                old_feature = F.normalize(old_feature, dim=1)

                phi_old = []
                phi_p = []
                for path in paths:
                    phi_old.append(old_feature_dict[path].view(1, -1))
                    phi_p.append(new_feature_dict[path].view(1, -1))
                phi_old = torch.cat(phi_old, dim=0).to(device)
                phi_p = torch.cat(phi_p, dim=0).to(device)

                loss = 1 - torch.mean(torch.sum(phi_p * feature, dim=1))
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                top1.update(acc1.item(), images.size(0))

                losses.update(loss.item(), images.size(0))

        return losses.avg, top1.avg


def build_feature_dict_transfer(loader: torch.utils.data.DataLoader,
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
            features = model(images)[-1].cpu()
        features = features.reshape(features.size(0), -1)
        features = F.normalize(features, dim=1)
        for path, feature in zip(paths, features):
            feature_dict[path] = feature

    return feature_dict