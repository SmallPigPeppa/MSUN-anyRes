#!/usr/bin/env bash
set -euo pipefail

# 1) define the exact order you want
models=(resnet50 densenet121 vgg16 mobilenetv2)
models=(resnet50 densenet121)

# 2) keep your dict of hyperparams
declare -A params=(
  [resnet50]="128:0.5:2e-5:90"
  [densenet121]="32:0.1:2e-5:90"
  [vgg16]="256:0.4:2e-5:90"
  [mobilenetv2]="32:0.1:2e-5:300"
)

# 3) iterate over the *ordered* list
for model in "${models[@]}"; do
  IFS=: read -r bs lr wd ep <<<"${params[$model]}"
  echo "Training $model (batch_size=$bs, lr=$lr, weight_decay=$wd, epochs=$ep)"
  python3 fixedres/main_sgd.py fit \
    --data.data_dir ./imagenet \
    --data.batch_size "$bs" \
    --data.num_workers 16 \
    --data.img_size 224 \
    --model.num_classes 1000 \
    --model.learning_rate "$lr" \
    --model.weight_decay "$wd" \
    --model.model_name "$model" \
    --trainer.max_epochs "$ep" \
    --trainer.precision bf16-mixed \
    --trainer.accelerator gpu \
    --trainer.logger WandbLogger \
    --trainer.logger.project msun-anyres \
    --trainer.logger.name "fixedres-$model" \
    --trainer.logger.log_model False \
    --trainer.logger.offline False \
    --model_checkpoint.dirpath "/mnt/bn/liuwenzhuo-lf/ckpt/msun/fixedres/$model" \
    --model_checkpoint.filename "epoch-{epoch:02d}-val_acc224-{val/acc224:.4f}" \
    --model_checkpoint.auto_insert_metric_name False \
    --model_checkpoint.monitor val/acc224 \
    --model_checkpoint.save_top_k 1 \
    --model_checkpoint.mode max \
    --model_checkpoint.save_last True \
    --lr_monitor.logging_interval epoch
done

