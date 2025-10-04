#!/bin/bash

DATASET_NAME="alzheimer_dataset"
DATA_PATH="data/${DATASET_NAME}"
OUTPUT_DIR="output/clip_alzheimer_ad"
CAP_DATA_PATH="./data/M3D_Cap_npy/M3D_Cap.json"


NUM_GPUS=2
FREE_GPUS=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
    | nl -v 0 \
    | sort -k2 -nr \
    | head -n $NUM_GPUS \
    | awk '{print $1}' \
    | paste -sd "," -)

export CUDA_VISIBLE_DEVICES=$FREE_GPUS
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"


deepspeed src/train/train_clip.py \
    --deepspeed ./scripts/zero2.json \
    --language_model_name_or_path medicalai/ClinicalBERT \
    --wb_name DCFormer_SigLIP \
    --vision_encoder "dcformer" \
    --loss_type "sigmoid" \
    --data_root ${DATA_PATH} \
    --cap_data_path ${CAP_DATA_PATH} \
    --max_length 128 \
    --bf16 False \
    --fp16 True \
    --output_dir ./output/DCFormer_SigLIP \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "no" \
    --eval_accumulation_steps 1 \
    --eval_steps 0.04 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 0.001 \
    --gradient_checkpointing False \
    --dataloader_pin_memory False \
    --dataloader_num_workers 4