#!/bin/bash
exp=zeju-0318-bs-1-vg-vicuna-index
# common_dataset=(ScanQA_v1.0_val_3dmit)
# common_dataset=(ScanNet ScanRefer ScanQA_multiplechoice )
common_dataset=(ScanRefer)
base_data_path=./src/data/3D_Benchmark
visfeat_type=local
token_num=256
layer=-2

target_dir=./ckpt/zeju_0318_bs_1_onlyvg_vicuna_index
answerdir=${target_dir}/answers-vg
mkdir -p ${answerdir}/${exp}
results_path=${target_dir}/results-vg
mkdir -p ${results_path}/${exp}


for dataset in ${common_dataset[*]}; do

    python inference_3d.py \
        --model 3dmit_peft \
        --encoder_pretrain epcl \
        --encoder_ckpt_path ./src/model_zoo/epcl_ckpts/epcl_scannet_vit-L-14_256tokens_latest.pth \
        --llm_ckpt_path ./src/model_zoo/vicuna-7b-v0\
        --delta_ckpt_path ${target_dir}/pytorch_model.pt \
        --max_tgt_len 1600 \
        --lora_r 32 \
        --lora_alpha 32 \
        --lora_dropout 0.1 \
        --vision_feature_type ${visfeat_type} \
        --num_vision_token ${token_num} \
        --vision_output_layer ${layer} \
        --conv_mode simple \
        --dataset-name ${dataset} \
        --base-data-path ${base_data_path} \
        --inference-mode common \
        --bs 1 \
        --answers-dir ${answerdir}/${exp} \
    
    # python common_eval_3d.py \
    #     --dataset-name ${dataset} \
    #     --answer-file ${answerdir}/${exp} \
    #     --base-data-path ${base_data_path} \
    #     2>&1 | tee ${results_path}/${exp}/eval_${dataset}.log
    python cal_scanqa_score.py \
        --dataset-name ${dataset} \
        --answer-file ${answerdir}/${exp} \
        --base-data-path ${base_data_path} \
        2>&1 | tee ${results_path}/${exp}/eval_${dataset}.log
done
