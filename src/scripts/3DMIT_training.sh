#!/bin/bash
numgpu=4
#zeju_0318_bs_1_onlyvg_vicuna_index
exp=111111
# dataname=VQA_74865_scannet_1218,VQA_74k_scannet_1225_xyzlocate
dataname=VG_36655_chat3dv2
visfeat_type=local
now=$(date +"%Y%m%d_%H%M%S")

ckpt_dir=../ckpt
mkdir -p ${ckpt_dir}/${exp}/log_rest/
deepspeed --include localhost:1,2,3,4 --master_addr 127.0.0.1 --master_port 28458 train.py \
    --stage 1 \
    --cfg ./src/config/train.yaml \
    --data_path  ./src/datasets/data/3D_Instruct/meta_file/${dataname}.json \
    --vision_root_path ./src/data/3D_Instruct/ \
    --max_tgt_len 1600 \
    --vision_type pcl \
    --use_system \
    --model 3dmit_peft \
    --encoder_pretrain epcl \
    --encoder_ckpt_path ./model_zoo/epcl_ckpts/epcl_scannet_vit-L-14_256tokens_latest.pth \
    --llm_ckpt_path ./src/model_zoo/vicuna-7b-v0\
    --vision_feature_type ${visfeat_type} \
    --num_vision_token 256 \
    --save_path  ${ckpt_dir}/${exp} \
    --log_path ${ckpt_dir}/${exp}/log_rest/ \
    2>&1 | tee ${ckpt_dir}/${exp}/log_rest/train_${now}.log
