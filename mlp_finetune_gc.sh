#! /bin/bash
eval "$(micromamba shell hook --shell=bash)"

nvidia-smi -q

export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline

seed=0
max_examples=204800
n_examples=$((2**$1*400))
n_partitions=$((max_examples/n_examples))
partition=$((seed%n_partitions))
batchsize=64
savedir=$SCRATCHBIG/saved_models_${seed}/baselines/${n_examples}
mkdir -p $savedir

datasetdir=$SCRATCHBIG/dataset
modeldir=$SCRATCHBIG/microsoft/codebert-base

if [[ ! -d $datasetdir ]]
then 
    bash copy_to_scratch.sh
fi

micromamba activate ML3.8

cd code || exit
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
    --num_train_epochs $((n_partitions*2)) \
    --num_train_examples $n_examples \
    --train_example_offset $((partition*n_examples)) \
    --block_size 256 \
    --train_batch_size $batchsize \
    --eval_batch_size 64 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --gradient_checkpointing \
    --early_stopping_patience $((n_partitions + 1)) \
    --seed $seed 2>&1| tee train.log


mkdir -p ./saved_models/
rm -rf ./saved_models/$lang
cp -r $savedir ./saved_models
