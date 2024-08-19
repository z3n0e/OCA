#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#

from accelerate import Accelerator
from utils.getters import get_model, get_optimizer
from utils.net_utils import LabelSmoothing
from dataset import SubImageCIFAR100
from trainers import BCTTrainer
import torch.nn as nn
import torch
from typing import Dict
from argparse import ArgumentParser
from collections import Counter
from utils.schedulers import get_policy

import yaml
from PIL import ImageFile
import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


def main(config: Dict) -> None:
    """Run training.

    :param config: A dictionary with all configurations to run training.
    :return:
    """

    torch.backends.cudnn.benchmark = True

    model = get_model(config.get('arch_params'))
    accelerator = Accelerator()
    device = accelerator.device

    old_model = get_model(config.get('old_arch_params'))
    old_model.load_state_dict(torch.load(config.get('old_model_path'))['model_state_dict'])

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model)
        old_model = torch.nn.DataParallel(old_model)
    model.to(device)
    old_model.to(device)

    alpha = float(config.get('alpha'))
    print(f'alpha is {alpha}')

    trainer = BCTTrainer()
    optimizer = get_optimizer(model, **config.get('optimizer_params'))
    data = SubImageCIFAR100(**config.get('dataset_params'))
    lr_policy = get_policy(optimizer, **config.get('lr_policy_params'))

    if config.get('label_smoothing') is None:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = LabelSmoothing(smoothing=config.get('label_smoothing'))

    optimizer, train_loader, val_loader =\
        accelerator.prepare(optimizer, data.train_loader, data.val_loader)
    old_model = accelerator.prepare(old_model)

    print("==>Preparing pesudo classifier")
    num_classes = int(config.get('arch_params')['num_classes'])
    embedding_dim = int(config.get('arch_params')['embedding_dim'])
    pseudo_classifier = torch.zeros(num_classes, embedding_dim)
    old_embedding_dim = embedding_dim
    if config.get('old_embedding_dim') is not None:
        old_embedding_dim = int(config.get('old_embedding_dim'))
    label_count = Counter()
    for i, (paths, (images, target)) in tqdm.tqdm(
        enumerate(train_loader), ascii=True, total=len(train_loader)
    ):
        images = images.to(device, non_blocking=True)
        target = target.cpu()
        with torch.no_grad():
            _, features = old_model(images)

        for feature, label in zip(features, target):
            pseudo_classifier[int(
                label)][:old_embedding_dim] += feature.flatten().cpu()
            label_count.update([int(label)])

    for i in range(num_classes):
        pseudo_classifier[i] = pseudo_classifier[i]/label_count[i]


    old_model = old_model.cpu()
    model = accelerator.prepare(model)
    # Training loop
    for epoch in range(config.get('epochs')):
        lr_policy(epoch, iteration=None)

        train_acc1, train_acc5, train_loss = trainer.train(
            train_loader=train_loader,
            model=model,
            alpha=alpha,
            pseudo_classifier=pseudo_classifier,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            accelerator=accelerator
        )

        print(
            "Train: epoch = {}, Loss = {}, Top 1 = {}, Top 5 = {}".format(
                epoch, train_loss, train_acc1, train_acc5
            ))

        test_acc1, test_acc5, test_loss = trainer.validate(
            val_loader=val_loader,
            model=model,
            alpha=alpha,
            pseudo_classifier=pseudo_classifier,
            criterion=criterion,
            device=device,
            accelerator=accelerator
        )

        print(
            "Test: epoch = {}, Loss = {}, Top 1 = {}, Top 5 = {}".format(
                epoch, test_loss, test_acc1, test_acc5
            ))
        
        if (epoch+1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, config.get('output_model_path'))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file for this pipeline.')
    args = parser.parse_args()
    with open(args.config) as f:
        read_config = yaml.safe_load(f)
    main(read_config)
