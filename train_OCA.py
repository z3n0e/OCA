from accelerate import Accelerator
from utils.schedulers import get_policy
from utils.getters import get_model, get_optimizer
from utils.net_utils import LabelSmoothing
from dataset import SubImageCIFAR100
from trainers import OCATrainer, build_feature_dict
from collections import Counter
import tqdm
import torch.nn as nn
import torch
from typing import Dict
from argparse import ArgumentParser
import torch.nn.functional as F

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


    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model)
        old_model = torch.nn.DataParallel(old_model)

    model.to(device)
    old_model.to(device)

    trainer = OCATrainer()
    optimizer = get_optimizer(model, **config.get('optimizer_params'))
    data = SubImageCIFAR100(**config.get('dataset_params'))
    lr_policy = get_policy(optimizer, **config.get('lr_policy_params'))
    train_loader = data.train_loader
    val_loader = data.val_loader

    optimizer, train_loader, val_loader =\
        accelerator.prepare(optimizer, train_loader, val_loader)

    if config.get('label_smoothing') is None:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = LabelSmoothing(smoothing=config.get('label_smoothing'))


    print("==>Preparing pesudo classifier and FeatureDict OLD")
    old_model = accelerator.prepare(old_model)
    old_model = old_model.eval().to(device)

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
            features = outputs[1]

        for feature, label in zip(features, target):
            pseudo_classifier[int(label)] += feature.flatten().cpu()
            label_count.update([int(label)])

    for i in range(num_classes):
        pseudo_classifier[i] = pseudo_classifier[i]/label_count[i]

    pseudo_classifier = pseudo_classifier.to(device)
    
    old_model = old_model.cpu()
    
    model = accelerator.prepare(model)
    print(':====>Training')
    # Training loop
    for epoch in range(config.get('epochs')):
        lr_policy(epoch, iteration=None)

        train_loss, train_loss_bct, train_loss_ce, train_loss_cosine = trainer.train(
            train_loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            accelerator=accelerator,
            pseudo_classifier=pseudo_classifier,
        )

        print(
            "Train: epoch = {}, Loss = {}, Loss_bct = {}, Loss_ce = {},Loss_cosine = {}".format(
                epoch, train_loss, train_loss_bct, train_loss_ce, train_loss_cosine
            ))

        test_loss, test_acc1 = trainer.validate(
            val_loader=data.val_loader,
            model=model,
            criterion=criterion,
            device=device
        )

        print(
            "Test: epoch = {}, Loss = {}, Top1 = {}".format(
                epoch, test_loss, test_acc1
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
