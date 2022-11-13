#!/bin/bash

python knowledge_distillation.py \
    --student_model /mnt/nlp4sd/mamooler/checkpoints/domain_adaptation/res/ledgar/ \
    --output_path /mnt/nlp4sd/mamooler/checkpoints/domain_adaptation/output/ledgar/ \
    --dataset_path /mnt/nlp4sd/mamooler/low-resource-text-classification-framework/lrtc_lib/data/available_datasets/ledgar/LEDGAR.txt \
    "$@"
