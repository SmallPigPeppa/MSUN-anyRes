#!/usr/bin/env bash
set -euo pipefail

# Define hyperparameters for each model in the format: batch_size:learning_rate:weight_decay:epochs
declare -A params=(
  [resnet50]="128:0.5:2e-5:90"
  [densenet121]="32:0.1:2e-5:90"
  [vgg16]="32:0.1:2e-5:90"
  [mobilenetv2]="32:0.1:2e-5:300"
)

for model in "${!params[@]}"; do
  # Split the parameter string into individual variables
  IFS=: read -r batch_size lr weight_decay max_epochs <<< "${params[$model]}"

  echo "Training $model (batch_size=$batch_size, lr=$lr, weight_decay=$weight_decay, epochs=$max_epochs)"

  python3 debug/main.py fit \
    --data.data_dir ./imagenet \
    --data.batch_size "$batch_size" \
    --data.num_workers 16 \
    --data.img_size 224 \
    --model.num_classes 1000 \
    --model.learning_rate "$lr" \
    --model.weight_decay "$weight_decay" \
    --model.model_name "$model" \
    --trainer.max_epochs "$max_epochs" \
    --trainer.precision bf16-mixed \
    --trainer.accelerator gpu \
    --trainer.logger WandbLogger \
    --trainer.logger.project msun-anyres \
    --trainer.logger.name "debug-fixedres-$model" \
    --trainer.logger.log_model False \
    --trainer.logger.offline False \
    --model_checkpoint.dirpath "/mnt/bn/liuwenzhuo-lf/ckpt/msun-anyres/fixedres/$model" \
    --model_checkpoint.monitor val/acc224 \
    --model_checkpoint.save_top_k 1 \
    --model_checkpoint.save_last True \
    --lr_monitor.logging_interval epoch
done
