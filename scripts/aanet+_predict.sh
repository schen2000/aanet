#!/usr/bin/env bash

# Predict

MODEL=pretrained/aanet+_sceneflow-d3e13ef0.pth
#MODEL=pretrained/aanet+_kitti15-2075aea1.pth
IMG_DIR=rund


CUDA_VISIBLE_DEVICES=0 python predict.py \
--data_dir rund/frms \
--pretrained_aanet $MODEL \
--feature_type ganet \
--save_type pfm \
--feature_pyramid \
--visualize \
--refinement_type hourglass \
--no_intermediate_supervision
