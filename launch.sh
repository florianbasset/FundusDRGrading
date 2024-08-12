#!/bin/bash

python src/fundusClassif/scripts/train.py --lr 0.00002935 --optimizer AdamW --ema 0 --swa 1 --as_regression 0 --data_augmentation_type heavy --mixup 0 --preprocessing clahe_rgb
python src/fundusClassif/scripts/train.py --lr 0.00002935 --optimizer AdamW --ema 0 --swa 1 --as_regression 0 --data_augmentation_type heavy --mixup 0 --preprocessing clahe_lab
python src/fundusClassif/scripts/train.py --lr 0.00002935 --optimizer AdamW --ema 0 --swa 1 --as_regression 0 --data_augmentation_type heavy --mixup 0 --preprocessing absent
python src/fundusClassif/scripts/train.py --lr 0.00002935 --optimizer AdamW --ema 0 --swa 1 --as_regression 0 --data_augmentation_type heavy --mixup 0 --preprocessing autobalance
python src/fundusClassif/scripts/train.py --lr 0.00002935 --optimizer AdamW --ema 0 --swa 1 --as_regression 0 --data_augmentation_type heavy --mixup 0 --preprocessing seoud
python src/fundusClassif/scripts/train.py --lr 0.00002935 --optimizer AdamW --ema 0 --swa 1 --as_regression 0 --data_augmentation_type heavy --mixup 0 --preprocessing sarki
python src/fundusClassif/scripts/train.py --lr 0.00002935 --optimizer AdamW --ema 0 --swa 1 --as_regression 0 --data_augmentation_type heavy --mixup 0 --preprocessing graham_meth1
python src/fundusClassif/scripts/train.py --lr 0.00002935 --optimizer AdamW --ema 0 --swa 1 --as_regression 0 --data_augmentation_type heavy --mixup 0 --preprocessing clahe_max_green_gsc
python src/fundusClassif/scripts/train.py --lr 0.00002935 --optimizer AdamW --ema 0 --swa 1 --as_regression 0 --data_augmentation_type heavy --mixup 0 --preprocessing graham_meth2




