#!/usr/bin/env bash
# Explicit startup script for training MultiScaleResNet via LightningCLI
# Model list
models=("resnet50" "densenet121" "vgg16" "mobilenetv2")
models=("resnet50")

for model_name in "${models[@]}"; do
  echo "Starting training for model: $model_name"

  python3 debug/main_adamw fit \
    --data.data_dir ./imagenet \
    --data.batch_size 256 \
    --data.num_workers 16 \
    --data.img_size 224 \
    --model.num_classes 1000 \
    --model.learning_rate 1e-3 \
    --model.weight_decay 1e-4 \
    --model.model_name "$model_name" \
    --trainer.max_epochs 80 \
    --trainer.precision bf16-mixed \
    --trainer.accelerator gpu \
    --trainer.logger WandbLogger \
    --trainer.logger.project msun-anyres \
    --trainer.logger.name "debug-adamw-fixedres-$model_name" \
    --trainer.logger.log_model False \
    --trainer.logger.offline False \
    --model_checkpoint.dirpath "/mnt/bn/liuwenzhuo-lf/ckpt/msun-anyres/fixedres/$model_name" \
    --model_checkpoint.monitor val/acc224 \
    --model_checkpoint.save_top_k 1 \
    --model_checkpoint.save_last True \
    --lr_monitor.logging_interval epoch

  echo "Finished training for model: $model_name"
done




