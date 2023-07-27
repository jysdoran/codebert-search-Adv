#!/bin/bash

# Grid Engine options (lines prefixed with #$)
# Runtime limit of 1 hour:
#$ -l h_rt=13:00:00
#
# Set working directory to the directory where the job is submitted from:
#$ -cwd
#
# Request one GPU: 
#$ -q gpu
#$ -pe gpu-a100 1
#
# Request 4 GB system RAM 
# the total system RAM available to the job is the value specified here multiplied by 
# the number of requested GPUs (above)
#$ -l h_vmem=100G
#$ -l h_rss=64G
#$ -l s_vmem=64G
#$ -l mem_free=64G

# . /etc/profile.d/modules.sh
# Initialaze micromamba ML3.8 environment
source $HOME/.bashrc
#micromamba activate $SCRATCH/micromamba/envs/ML3.8
conda activate CondaML3.8

# Run the executable
lang=python

seed=0
max_examples=$((2**9*400))
n_examples=$((2**$1*400))
n_partitions=$((max_examples/n_examples))
partition=$((seed%n_partitions))

datasetdir=$SCRATCH/CodeXGLUE/Text-Code/NL-code-search-Adv/dataset

cd code
output_dir=./baselines/${n_examples}_${seed}
mkdir -p $output_dir

python run.py \
    --output_dir=$output_dir \
    --model_type=roberta \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=roberta-base \
    --do_train \
    --do_eval \
    --do_test \
    --train_data_file=$datasetdir/train.jsonl \
    --eval_data_file=$datasetdir/valid.jsonl \
    --test_data_file=$datasetdir/test.jsonl \
    --num_train_epochs $((n_partitions*2)) \
    --num_train_examples $n_examples \
    --train_example_offset $((partition*n_examples)) \
    --block_size 256 \
    --train_batch_size 64 \
    --eval_batch_size 128 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed $seed 2>&1| tee train.log
#    --gradient_checkpointing \
