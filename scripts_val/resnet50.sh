#!/usr/bin/env bash

python3 msun/resnet50_demo.py test \
  --data.data_dir     ./imagenet \
  --data.batch_size   256 \
  --data.num_workers  16 \
  --data.img_size     224 \
  --model.num_classes 1000 \
  --trainer.accelerator    gpu \
  --trainer.precision      bf16-mixed \
  --trainer.logger         WandbLogger \
  --trainer.logger.project msun-anyres \
  --trainer.logger.name    msun-resnet50-test \
  --trainer.logger.offline False \
  --ckpt_path /mnt/bn/liuwenzhuo-lf/ckpt/msun/msun/resnet50/last.ckpt
