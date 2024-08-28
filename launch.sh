#!/bin/bash

python src/fundusClassif/scripts/train.py --lr 0.00025 --preprocessing sarki --model efficientnet_b0.ra_in1k
python src/fundusClassif/scripts/train.py --lr 0.000025 --preprocessing seoud --model efficientnet_b0.ra_in1k
python src/fundusClassif/scripts/train.py --lr 0.00005 --preprocessing autobalance --model efficientnet_b0.ra_in1k
python src/fundusClassif/scripts/train.py --lr 0.00005 --preprocessing clahe_max_green_gsc --model efficientnet_b0.ra_in1k
python src/fundusClassif/scripts/train.py --lr 0.00025 --preprocessing clahe_rgb --model efficientnet_b0.ra_in1k
python src/fundusClassif/scripts/train.py --lr 0.00005 --preprocessing clahe_lab --model efficientnet_b0.ra_in1k
python src/fundusClassif/scripts/train.py --lr 0.00005 --preprocessing absent --model efficientnet_b0.ra_in1k

python src/fundusClassif/scripts/train.py --lr 0.00025 --preprocessing sarki --model convnextv2_nano.fcmae_ft_in22k_in1k_384
python src/fundusClassif/scripts/train.py --lr 0.000025 --preprocessing seoud --model convnextv2_nano.fcmae_ft_in22k_in1k_384
python src/fundusClassif/scripts/train.py --lr 0.00005 --preprocessing autobalance --model convnextv2_nano.fcmae_ft_in22k_in1k_384
python src/fundusClassif/scripts/train.py --lr 0.00005 --preprocessing clahe_max_green_gsc --model convnextv2_nano.fcmae_ft_in22k_in1k_384
python src/fundusClassif/scripts/train.py --lr 0.00025 --preprocessing clahe_rgb --model convnextv2_nano.fcmae_ft_in22k_in1k_384
python src/fundusClassif/scripts/train.py --lr 0.00005 --preprocessing clahe_lab --model convnextv2_nano.fcmae_ft_in22k_in1k_384
python src/fundusClassif/scripts/train.py --lr 0.00005 --preprocessing absent --model convnextv2_nano.fcmae_ft_in22k_in1k_384

python src/fundusClassif/scripts/train.py --lr 0.00025 --preprocessing sarki --model resnetv2_50.a1h_in1k
python src/fundusClassif/scripts/train.py --lr 0.000025 --preprocessing seoud --model resnetv2_50.a1h_in1k
python src/fundusClassif/scripts/train.py --lr 0.00005 --preprocessing autobalance --model resnetv2_50.a1h_in1k
python src/fundusClassif/scripts/train.py --lr 0.00005 --preprocessing clahe_max_green_gsc --model resnetv2_50.a1h_in1k
python src/fundusClassif/scripts/train.py --lr 0.00025 --preprocessing clahe_rgb --model resnetv2_50.a1h_in1k
python src/fundusClassif/scripts/train.py --lr 0.00005 --preprocessing clahe_lab --model resnetv2_50.a1h_in1k
python src/fundusClassif/scripts/train.py --lr 0.00005 --preprocessing absent --model resnetv2_50.a1h_in1k

python src/fundusClassif/scripts/train.py --lr 0.00025 --preprocessing sarki --model vit_base_patch16_224.dino
python src/fundusClassif/scripts/train.py --lr 0.000025 --preprocessing seoud --model vit_base_patch16_224.dino
python src/fundusClassif/scripts/train.py --lr 0.00005 --preprocessing autobalance --model vit_base_patch16_224.dino
python src/fundusClassif/scripts/train.py --lr 0.00005 --preprocessing clahe_max_green_gsc --model vit_base_patch16_224.dino
python src/fundusClassif/scripts/train.py --lr 0.00025 --preprocessing clahe_rgb --model vit_base_patch16_224.dino
python src/fundusClassif/scripts/train.py --lr 0.00005 --preprocessing clahe_lab --model vit_base_patch16_224.dino
python src/fundusClassif/scripts/train.py --lr 0.00005 --preprocessing absent --model vit_base_patch16_224.dino
