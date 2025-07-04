#!/usr/bin/env bash
# Explicit startup script for training MultiScaleResNet via LightningCLI

# Example: modify values as needed
python3 msun/resnet50.py \
  --data_dir /path/to/imagenet \
  --batch_size 256 \
  --num_workers 8 \
  --img_size 224 \
  --num_classes 1000 \
  --learning_rate 1e-3 \
  --weight_decay 1e-4 \
  --max_epochs 100 \
  --unified_res 56 \
  --alpha 1.0 \
  --gpus 1 \
  --accelerator ddp \
  --run_name msun_experiment \
  --model_checkpoint.dirpath ./checkpoints \
  --model_checkpoint.monitor val/acc224 \
  --model_checkpoint.save_top_k 1 \
  --model_checkpoint.save_last True \
  --lr_monitor.logging_interval epoch