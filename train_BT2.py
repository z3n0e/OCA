#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#

from accelerate import Accelerator
from utils.schedulers import get_policy
from utils.getters import get_model, get_optimizer
from dataset import SubImageCIFAR100
from trainers import TransferTrainer, build_feature_dict_transfer
from collections import Counter
import tqdm
import torch.nn as nn
import torch
from typing import Dict
from argparse import ArgumentParser
import yaml
from PIL import ImageFile

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

    new_model = get_model(config.get('new_arch_params'))
    new_model.load_state_dict(torch.load(config.get('new_model_path'))['model_state_dict'])

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model)
        old_model = torch.nn.DataParallel(old_model)
        new_model = torch.nn.DataParallel(new_model)

    trainer = TransferTrainer()
    optimizer = get_optimizer(model, **config.get('optimizer_params'))
    data = SubImageCIFAR100(**config.get('dataset_params'))

    lr_policy = get_policy(optimizer, **config.get('lr_policy_params'))
    lambda_1 = float(config.get('lambda_1'))
    lambda_2 = float(config.get('lambda_2'))
    lambda_3 = float(config.get('lambda_3'))
    train_loader = data.train_loader
    val_loader = data.val_loader

    optimizer, train_loader, val_loader =\
        accelerator.prepare(optimizer, train_loader, val_loader)

    criterion = nn.CosineSimilarity(dim=1)

    print("==>Preparing pesudo classifier")
    old_model = accelerator.prepare(old_model)
    num_classes = int(config.get('arch_params')['num_classes'])
    embedding_dim = int(config.get('arch_params')['embedding_dim_old'])
    pseudo_classifier = torch.zeros(
        num_classes, embedding_dim, requires_grad=False)
    label_count = Counter()
    for i, (paths, (images, target)) in tqdm.tqdm(
        enumerate(data.train_loader), ascii=True, total=len(data.train_loader)
    ):
        images = images.to(device, non_blocking=True)
        target = target.cpu()
        with torch.no_grad():
            outputs = old_model(images)
            features = outputs[-1]

        for feature, label in zip(features, target):
            pseudo_classifier[int(label)] += feature.flatten().cpu()
            label_count.update([int(label)])

    for i in range(num_classes):
        pseudo_classifier[i] = pseudo_classifier[i]/label_count[i]
    old_model = old_model.cpu()


    pseudo_classifier = pseudo_classifier.to(device)

    print(':====>BuildFeatureDict')
    # build a dictionary of saved features, so that don't need to recompute
    # old features at each iteration
    old_feature_dict = {}
    new_feature_dict = {}

    new_model = accelerator.prepare(new_model)
    new_feature_dict = build_feature_dict_transfer(
        train_loader, new_model, device, new_feature_dict)
    new_feature_dict = build_feature_dict_transfer(
        val_loader, new_model, device, new_feature_dict)
    new_model = new_model.cpu()

    old_model = accelerator.prepare(old_model)
    old_feature_dict = build_feature_dict_transfer(
        train_loader, old_model, device, old_feature_dict)
    old_feature_dict = build_feature_dict_transfer(
        val_loader, old_model, device, old_feature_dict)
    old_model = old_model.cpu()

    model = accelerator.prepare(model)
    print(':====>Training')
    # Training loop
    for epoch in range(config.get('epochs')):
        lr_policy(epoch, iteration=None)

        train_loss = trainer.train(
            train_loader=train_loader,
            model=model,
            old_feature_dict=old_feature_dict,
            new_feature_dict=new_feature_dict,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            accelerator=accelerator,
            pseudo_classifier=pseudo_classifier,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            lambda_3=lambda_3,
        )

        print(
            "Train: epoch = {}, Loss = {}".format(
                epoch, train_loss
            ))

        test_loss, test_acc1 = trainer.validate(
            val_loader=data.val_loader,
            model=model,
            old_feature_dict=old_feature_dict,
            new_feature_dict=new_feature_dict,
            criterion=criterion,
            device=device
        )
        
        print(
            "Test: epoch = {}, Loss = {}, Top1 = {}".format(
                epoch, test_loss, test_acc1
            ))

        # make checkpoints
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
