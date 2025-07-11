#!/bin/bash
# global batch size = 32*8 = 256
# 32: 每张卡87%; 16: 每张卡79%
NCCL_P2P_DISABLE=1 \
deepspeed --include localhost:0,7 llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /home/liusn/02resources/02ckpt/vicuna-7b-v1.5 \
    --version plain \
    --data_path /home/liusn/02resources/00dataset/Quilt-LLaVA-Pretrain/quilt_pretrain.json \
    --image_folder /home/liusn/02resources/00dataset/Quilt-LLaVA-Pretrain/quilt_1m \
    --vision_tower /repository/users/liusn/02ckpt/QuiltNet-B-32 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /home/liusn/02resources/01output/qllava_wsi_output/qllava-debug \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 800 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --mm_vision_select_feature "cls_patch" \
    --stage 1 \
    --report_to wandb \
    > /home/liusn/02resources/01output/qllava_wsi_output/qllava-debug/qllava-debug-pretrain-0424.log 2>&1 &
