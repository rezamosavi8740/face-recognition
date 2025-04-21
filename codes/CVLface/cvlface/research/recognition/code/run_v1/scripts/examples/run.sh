#!/bin/bash

LIGHTING_TESTING=1 CUDA_VISIBLE_DEVICES=0 python train.py \
  trainers=configs/default.yaml \
  trainers.num_gpu=1 \
  trainers.batch_size=512 \
  trainers.gradient_acc=1 \
  trainers.num_workers=16 \
  trainers.precision='32-true' \
  trainers.float32_matmul_precision='high' \
  dataset=configs/400K.yaml \
  data_augs=configs/basic_v1.yaml \
  models=iresnet/configs/v1_ir101.yaml \
  pipelines=configs/train_model_cls.yaml \
  evaluations=configs/quick.yaml \
  classifiers=configs/fc.yaml \
  optims=configs/cosine.yaml \
  losses=configs/adaface.yaml \
  trainers.output_dir="/home/user1/temp/checkpoints" \
  +trainers.max_epochs=5 \
  trainers.skip_final_eval=False