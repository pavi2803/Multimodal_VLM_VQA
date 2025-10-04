#!/bin/bash

# python src/eval/eval_clip.py \
#     --model_name_or_path output/DCFormer_SigLIP \
#     --data_root ./data \
#     --save_output True \
#     --output_dir ./output/eval \
#     --max_length 512
    
    

    
DATASET_NAME="alzheimer_dataset"
DATA_PATH="data/${DATASET_NAME}"
OUTPUT_DIR="output/clip_alzheimer_ad"
CAP_DATA_PATH="./data/M3D_Cap_npy/M3D_Cap.json"


# python src/eval/eval_clip.py \
#     --model_name_or_path output/DCFormer_SigLIP \
#     --data_root ${DATA_PATH} \
#     --cap_data_path ${CAP_DATA_PATH}
#     --save_output True \
#     --output_dir ./output/eval \
#     --max_length 128



FREE_GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
           | sort -k2 -nr | head -n1 | awk '{print $1}')

echo "Using GPU: $FREE_GPU"

CUDA_VISIBLE_DEVICES=$FREE_GPU python src/eval/eval_clip.py \
    --model_name_or_path output/DCFormer_SigLIP \
    --data_root ${DATA_PATH} \
    --cap_data_path ${CAP_DATA_PATH} \
    --save_output True \
    --output_dir ./output/eval \
    --max_length 128