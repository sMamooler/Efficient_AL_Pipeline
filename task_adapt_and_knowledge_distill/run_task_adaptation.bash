#!/bin/bash


python task_adaptation.py \
    --model_name_or_path roberta-base \
    --train_file /mnt/nlp4sd/mamooler/low-resource-text-classification-framework/lrtc_lib/data/available_datasets/ledgar/LEDGAR.txt \
    --output_dir /mnt/nlp4sd/mamooler/checkpoints/domain_adaptation/res/ledgar \
    --num_train_epochs 5 \
    --per_device_train_batch_size 64 \
    --learning_rate 3e-4 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model loss \
    --load_best_model_at_end \
    --save_steps 10000 \
    --eval_steps 10000 \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --fp16 \
    "$@"
