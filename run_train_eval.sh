#!/bin/bash

python train_independent.py --config configs/cifar100_independent_old.yaml
python train_independent.py --config configs/cifar100_independent_new.yaml

python train_BCT.py --config configs/cifar100_BCT.yaml

python train_OCA.py --config configs/train/cifar100_OCA.yaml

python train_BT2.py --config configs/cifar100_BT2.yaml


python eval.py --config configs/cifar100_eval_old_old.yaml
python eval.py --config configs/cifar100_eval_old_new.yaml
python eval.py --config configs/cifar100_eval_new_new.yaml

python eval.py --config configs/cifar100_eval_old_new_BCT.yaml
python eval.py --config configs/cifar100_eval_new_new_BCT.yaml

python eval.py --config configs/cifar100_eval_new_new_OCA.yaml
python eval.py --config configs/cifar100_eval_old_new_OCA.yaml

python eval.py --config configs/cifar100_eval_old_new_BT2.yaml
python eval.py --config configs/cifar100_eval_new_new_BT2.yaml