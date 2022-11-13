#!/bin/bash


python utils.py \
    --model_path /mnt/nlp4sd/mamooler/checkpoints/domain_adaptation/output/contract-nli/model-distillation-10epochs-2022-06-18_14-11-39 \
    --method distilled \
    --data_dir ../data/available_datasets/contract_nli/train.csv \
    --meta_data_dir ../data/available_datasets/contract_nli/train_meta_data.csv \
    --output_dir contract_nli/normalized/ \
    --avg_cluster_size 10 \
    "$@"
