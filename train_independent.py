#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#

from typing import Dict
from argparse import ArgumentParser

import yaml
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn

from trainers import BackboneTrainer
from dataset import SubImageCIFAR100
from utils.net_utils import LabelSmoothing
from utils.getters import get_model, get_optimizer
from utils.schedulers import get_policy
from accelerate import Accelerator


def main(config: Dict) -> None:
    """Run training.

    :param config: A dictionary with all configurations to run training.
    :return:
    """

    torch.backends.cudnn.benchmark = True

    model = get_model(config.get('arch_params'))
    accelerator = Accelerator()
    device = accelerator.device

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model)
    model.to(device)

    trainer = BackboneTrainer()
    optimizer = get_optimizer(model, **config.get('optimizer_params'))
    data = SubImageCIFAR100(**config.get('dataset_params'))
    lr_policy = get_policy(optimizer, **config.get('lr_policy_params'))


    if config.get('label_smoothing') is None:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = LabelSmoothing(smoothing=config.get('label_smoothing'))

    train_loader = data.train_loader
    val_loader = data.val_loader

    # Training loop
    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, \
        train_loader, val_loader)
        
    for epoch in range(config.get('epochs')):
        lr_policy(epoch, iteration=None)

        train_acc1, train_acc5, train_loss = trainer.train(
            train_loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            accelerator = accelerator
        )

        print(
            "Train: epoch = {}, Loss = {}, Top 1 = {}, Top 5 = {}".format(
                epoch, train_loss, train_acc1, train_acc5
            ))

        if (epoch+1) % 1 == 0:
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
            }, config.get('output_model_path'))

        model = accelerator.prepare(model)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file for this pipeline.')
    args = parser.parse_args()
    with open(args.config) as f:
        read_config = yaml.safe_load(f)
    main(read_config)
