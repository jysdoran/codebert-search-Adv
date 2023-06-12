#! /bin/bash
eval "$(micromamba shell hook --shell=bash)"

nvidia-smi -q

batchsize=32
lang=python_ddp_$batchsize
#datasetdir=$HF_HOME/datasets/downloads/extracted/12f42e6cc1af7629398f6c47e7c7be2e772d82061287e04771732698b8e0e110/$lang/final/jsonl
savedir=$SCRATCHBIG/saved_models/$lang
mkdir -p $savedir
#datasetdir=$SCRATCH/CodeXGLUE/Text-Code/NL-code-search-Adv/dataset
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline

datasetdir=$SCRATCHBIG/dataset
modeldir=$SCRATCHBIG/microsoft/codebert-base

if [[ ! -d $datasetdir ]]
then 
    bash copy_to_scratch.sh
fi

micromamba activate ML3.8

cd code
python run.py \
    --output_dir=$savedir \
    --model_type=roberta \
    --config_name=$modeldir \
    --model_name_or_path=$modeldir \
    --tokenizer_name=$modeldir \
    --do_train \
    --do_eval \
    --do_test \
    --train_data_file=$datasetdir/train.jsonl \
    --eval_data_file=$datasetdir/valid.jsonl \
    --test_data_file=$datasetdir/test.jsonl \
    --num_train_epochs 2 \
    --block_size 256 \
    --train_batch_size $batchsize \
    --eval_batch_size 64 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --gpu_batch_contrasting \
    --seed 123456 2>&1| tee train.log

    # --tokenizer_name=roberta-base \
    # --train_batch_size 32 \
    # --train_data_file=../dataset/train.jsonl \
    # --eval_data_file=../dataset/valid.jsonl \
    # --test_data_file=../dataset/test.jsonl \
    # --train_data_file=$datasetdir/train/${lang}_train_0.jsonl.gz \
    # --eval_data_file=$datasetdir/valid/${lang}_valid_0.jsonl.gz \
    # --test_data_file=$datasetdir/test/${lang}_test_0.jsonl.gz \
mkdir -p ./saved_models/
rm -rf ./saved_models/$lang
cp -r $savedir ./saved_models
