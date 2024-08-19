#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#

from typing import Dict
from argparse import ArgumentParser

import torch
import yaml
from utils.getters import get_model
import json

from dataset import SubImageFolder, SubImageCIFAR100
from utils.eval_utils import cmc_evaluate


def main(config: Dict) -> None:
    """Run evaluation.

    :param config: A dictionary with all configurations to run evaluation.
    """
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"

    # Load models:
    if config.get('arch_params') is not None:
        model = get_model(config.get('arch_params'))
        model.load_state_dict(torch.load(config.get('query_model_path'))['model_state_dict'])
        query_model = model.to(device)

    gallery_model = query_model
    if config.get('gallery_arch_params') is not None:
        gallery_model = get_model(config.get('gallery_arch_params'))
        gallery_model.load_state_dict(torch.load(config.get('gallery_model_path'))['model_state_dict'])

    if isinstance(gallery_model, torch.nn.DataParallel):
        gallery_model = gallery_model.module
    if isinstance(query_model, torch.nn.DataParallel):
        query_model = query_model.module
    data = SubImageCIFAR100(**config.get('dataset_params'))

    val_loader = [value for i, value in enumerate(
        data.val_loader) if i % 1 == 0]

    cmc_out, mean_ap_out = cmc_evaluate(
        gallery_model,
        query_model,
        val_loader,
        device,
        **config.get('eval_params')
    )

    if config.get('eval_params').get('per_class'):
        print('CMC Top-1 = {}, CMC Top-5 = {}'.format(*cmc_out[0]))
        print('Per class CMC: {}'.format(cmc_out[1]))
    else:
        print('CMC Top-1 = {}, CMC Top-5 = {}'.format(*cmc_out))

    if mean_ap_out is not None:
        print('mAP = {}'.format(mean_ap_out))

    with open(config.get('txt_log_path'), 'w') as file:
        # Write content to the file
        file.write('CMC Top-1 = {}, CMC Top-5 = {}\n'.format(*cmc_out))
        file.write('mAP = {}'.format(mean_ap_out))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file for this pipeline.')
    args = parser.parse_args()
    with open(args.config) as f:
        read_config = yaml.safe_load(f)
    main(read_config)
