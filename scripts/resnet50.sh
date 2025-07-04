#!/usr/bin/env bash
# Explicit startup script for training MultiScaleResNet via LightningCLI

# Example: modify values as needed
python3 msun/resnet50.py fit \
  --data.data_dir /mnt/hdfs/byte_content_security/user/liuwenzhuo/datasets/parquet/imagenet \
  --data.batch_size 128 \
  --data.num_workers 8 \
  --data.img_size 224 \
  --model.num_classes 1000 \
  --model.learning_rate 1e-3 \
  --model.weight_decay 1e-4 \
  --model.unified_res 56 \
  --model.alpha 1.0 \
  --trainer.max_epochs 90 \
  --trainer.devices 8 \
  --trainer.precision 16 \
  --trainer.accelerator gpu \
  --trainer.logger WandbLogger \
  --trainer.logger.project msun-anyres \
  --trainer.logger.name msun-RN50 \
  --trainer.logger.log_model False \
  --trainer.logger.offline False \
  --model_checkpoint.dirpath ./checkpoints \
  --model_checkpoint.monitor val/acc224 \
  --model_checkpoint.save_top_k 1 \
  --model_checkpoint.save_last True \
  --lr_monitor.logging_interval epoch

#  --trainer.strategy ddp_find_unused_parameters_true \
