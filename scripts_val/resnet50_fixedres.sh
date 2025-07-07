#!/usr/bin/env bash
# Test multiple models with their respective checkpoints

models=(resnet50 densenet121 vgg16 mobilenetv2)
declare -A ckpt_paths=(
  [resnet50]="/mnt/bn/liuwenzhuo-lf/ckpt/msun/fixedres/resnet50/last.ckpt"
  [densenet121]="/mnt/bn/liuwenzhuo-lf/ckpt/msun/fixedres/densenet121/last.ckpt"
  [vgg16]="/mnt/bn/liuwenzhuo-lf/ckpt/msun/fixedres/vgg16/last.ckpt"
  [mobilenetv2]="/mnt/bn/liuwenzhuo-lf/ckpt/msun/fixedres/mobilenetv2/last.ckpt"
)

for m in "${models[@]}"; do
  # run test for model $m
  python3 fixedres/main_sgd.py test \
    --data.data_dir     ./imagenet \
    --data.batch_size   256 \
    --data.num_workers  16 \
    --data.img_size     224 \
    --model.num_classes 1000 \
    --model.model_name  "$m" \
    --trainer.accelerator    gpu \
    --trainer.precision      bf16-mixed \
    --trainer.logger         WandbLogger \
    --trainer.logger.project msun-anyres-val \
    --trainer.logger.name    fixedres-"$m" \
    --trainer.logger.offline False \
    --ckpt_path            "${ckpt_paths[$m]}"
done
