#!/bin/bash

# Grid Engine options (lines prefixed with #$)
# Runtime limit of 1 hour:
#$ -l h_rt=13:00:00
#
# Set working directory to the directory where the job is submitted from:
#$ -cwd
#
# Request one GPU: 
#$ -pe gpu 1
#
# Request 4 GB system RAM 
# the total system RAM available to the job is the value specified here multiplied by 
# the number of requested GPUs (above)
#$ -l h_vmem=20G

# Initialise the environment modules and load CUDA version 8.0.61
# . /etc/profile.d/modules.sh
# Initialaze micromamba ML3.8 environment
mambaml

# Run the executable
lang=python
mkdir -p ./saved_models/$lang
datasetdir=$SCRATCH/CodeXGLUE/Text-Code/NL-code-search-Adv/dataset

cd code
python run.py \
    --output_dir=./saved_models/$lang \
    --model_type=roberta \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=roberta-base \
    --do_train \
    --train_data_file=$datasetdir/train.jsonl \
    --eval_data_file=$datasetdir/valid.jsonl \
    --test_data_file=$datasetdir/test.jsonl \
    --num_train_epochs 2 \
    --block_size 256 \
    --train_batch_size $1 \
    --eval_batch_size 128 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --gradient_checkpointing \
    --seed 123456 2>&1| tee train.log