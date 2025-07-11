#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/home/liusn/01codes/03pathology/quilt-llava

mkdir -p /home/liusn/02resources/01output/qllava_wsi_output/qllava-lora-0516

#    --model_name_or_path /repository/users/liusn/02ckpt/quilt-llava-ckpt2/Quilt-LLaVA-fc_conl/llava-clip-loss-pretrain-0425 \
#    --model_base_path /home/liusn/02resources/02ckpt/vicuna-7b-v1.5 \
NCCL_P2P_DISABLE=1 \
deepspeed --include localhost:0 llava/train/train_mem.py \
     --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero2.json \
    --pretrain_mm_mlp_adapter /repository/users/liusn/02ckpt/quilt-llava-ckpt2/Quilt-LLaVA-fc_conl/llava-clip-loss-pretrain-0425/mm_projector.bin \
    --model_name_or_path /home/liusn/02resources/02ckpt/vicuna-7b-v1.5 \
    --version v1 \
    --data_path /home/liusn/02resources/00dataset/WsiVQA/WsiVQA_quilt_train.json \
    --image_folder /home/liusn/02resources/00dataset/TCGA-BRCA-patient-feats \
    --vision_tower /repository/users/liusn/02ckpt/QuiltNet-B-32 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --freeze_backbone True \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir /home/liusn/02resources/01output/qllava_wsi_output/qllava-lora-0516 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 128 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 600 \
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
    --stage 3 \
    --report_to wandb \
    > /home/liusn/02resources/01output/qllava_wsi_output/qllava-lora-0516/qllava-wsi-output-0516.log 2>&1 &
