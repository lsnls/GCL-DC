#!/bin/bash
# global batch size = 16*8 = 128
# 32: 单张卡95%
NCCL_P2P_DISABLE=1 \
deepspeed --include localhost:0,5 llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /mnt/f70f6709-366e-49a0-861f-497645d98975/liusn/02ckpt/vicuna-7b-v1.5 \
    --version v1 \
    --data_path /mnt/f70f6709-366e-49a0-861f-497645d98975/liusn/00dataset/QUILT-LLaVA-Instruct-107K/quilt_instruct_67k.json \
    --image_folder /mnt/f70f6709-366e-49a0-861f-497645d98975/liusn/00dataset/QUILT-LLaVA-Instruct-107K/quilt_instruct \
    --vision_tower /mnt/f70f6709-366e-49a0-861f-497645d98975/liusn/02ckpt/QuiltNet-B-32 \
    --pretrain_mm_mlp_adapter /mnt/f70f6709-366e-49a0-861f-497645d98975/liusn/02ckpt/Quilt-LLaVA-fc_conl/llava-clip-loss-pretrain-0425/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir /mnt/f70f6709-366e-49a0-861f-497645d98975/liusn/02ckpt/Quilt-LLaVA-fc_conl/llava-lm-loss-subset-lora-0731 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 300 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
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
    --stage 2 \
    --report_to wandb \
    > /mnt/f70f6709-366e-49a0-861f-497645d98975/liusn/01output/Quilt-LLaVA-fc_conl/llava-lm-loss-subset-lora-0731.log 2>&1 &
