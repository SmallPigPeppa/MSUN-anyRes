#!/usr/bin/env bash
# Explicit startup script for training MultiScaleResNet via LightningCLI

# Example: modify values as needed
python3 msun/densenet121.py fit \
  --data.data_dir ./imagenet \
  --data.batch_size 256 \
  --data.num_workers 16 \
  --data.img_size 224 \
  --model.num_classes 1000 \
  --model.learning_rate 1e-3 \
  --model.weight_decay 1e-4 \
  --model.alpha 1.0 \
  --trainer.max_epochs 90 \
  --trainer.precision bf16-mixed \
  --trainer.accelerator gpu \
  --trainer.logger WandbLogger \
  --trainer.logger.project msun-anyres \
  --trainer.logger.name msun-densenet121 \
  --trainer.logger.log_model False \
  --trainer.logger.offline False \
  --model_checkpoint.dirpath /mnt/bn/liuwenzhuo-lf/ckpt/msun-anyres/densenet121 \
  --model_checkpoint.monitor val/acc224 \
  --model_checkpoint.save_top_k 1 \
  --model_checkpoint.save_last True \
  --lr_monitor.logging_interval epoch


