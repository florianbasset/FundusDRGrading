#!/bin/bash

python src/fundusClassif/scripts/train.py --lr 0.0005 --preprocessing sarki --model convnextv2_base.fcmae_ft_in1k
python src/fundusClassif/scripts/train.py --lr 0.00005 --preprocessing seoud --model convnextv2_base.fcmae_ft_in1k
python src/fundusClassif/scripts/train.py --lr 0.0001 --preprocessing autobalance --model convnextv2_base.fcmae_ft_in1k
python src/fundusClassif/scripts/train.py --lr 0.0001 --preprocessing clahe_max_green_gsc --model convnextv2_base.fcmae_ft_in1k
python src/fundusClassif/scripts/train.py --lr 0.0005 --preprocessing clahe_rgb --model convnextv2_base.fcmae_ft_in1k
python src/fundusClassif/scripts/train.py --lr 0.0001 --preprocessing clahe_lab --model convnextv2_base.fcmae_ft_in1k
python src/fundusClassif/scripts/train.py --lr 0.0001 --preprocessing absent --model convnextv2_base.fcmae_ft_in1k

python src/fundusClassif/scripts/train.py --lr 0.0005 --preprocessing sarki --model efficientnetv2_rw_m.agc_in1k
python src/fundusClassif/scripts/train.py --lr 0.00005 --preprocessing seoud --model efficientnetv2_rw_m.agc_in1k
python src/fundusClassif/scripts/train.py --lr 0.0001 --preprocessing autobalance --model efficientnetv2_rw_m.agc_in1k
python src/fundusClassif/scripts/train.py --lr 0.0001 --preprocessing clahe_max_green_gsc --model efficientnetv2_rw_m.agc_in1k
python src/fundusClassif/scripts/train.py --lr 0.0005 --preprocessing clahe_rgb --model efficientnetv2_rw_m.agc_in1k
python src/fundusClassif/scripts/train.py --lr 0.0001 --preprocessing clahe_lab --model efficientnetv2_rw_m.agc_in1k
python src/fundusClassif/scripts/train.py --lr 0.0001 --preprocessing absent --model efficientnetv2_rw_m.agc_in1k


python src/fundusClassif/scripts/train.py --lr 0.0005 --preprocessing sarki --model resnetv2_50.a1h_in1k
python src/fundusClassif/scripts/train.py --lr 0.00005 --preprocessing seoud --model resnetv2_50.a1h_in1k
python src/fundusClassif/scripts/train.py --lr 0.0001 --preprocessing autobalance --model resnetv2_50.a1h_in1k
python src/fundusClassif/scripts/train.py --lr 0.0001 --preprocessing clahe_max_green_gsc --model resnetv2_50.a1h_in1k
python src/fundusClassif/scripts/train.py --lr 0.0005 --preprocessing clahe_rgb --model resnetv2_50.a1h_in1k
python src/fundusClassif/scripts/train.py --lr 0.0001 --preprocessing clahe_lab --model resnetv2_50.a1h_in1k
python src/fundusClassif/scripts/train.py --lr 0.0001 --preprocessing absent --model resnetv2_50.a1h_in1k


python src/fundusClassif/scripts/train.py --lr 0.0005 --preprocessing sarki --model vit_base_patch16_224.dino
python src/fundusClassif/scripts/train.py --lr 0.00005 --preprocessing seoud --model vit_base_patch16_224.dino
python src/fundusClassif/scripts/train.py --lr 0.0001 --preprocessing autobalance --model vit_base_patch16_224.dino
python src/fundusClassif/scripts/train.py --lr 0.0001 --preprocessing clahe_max_green_gsc --model vit_base_patch16_224.dino
python src/fundusClassif/scripts/train.py --lr 0.0005 --preprocessing clahe_rgb --model vit_base_patch16_224.dino
python src/fundusClassif/scripts/train.py --lr 0.0001 --preprocessing clahe_lab --model vit_base_patch16_224.dino
python src/fundusClassif/scripts/train.py --lr 0.0001 --preprocessing absent --model vit_base_patch16_224.dino
