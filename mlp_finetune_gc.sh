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
model_path=saved_models_${batchsize}_${seed}/baselines/${n_examples}
savedir=$SCRATCHBIG/$model_path
mkdir -p $savedir

datasetdir=$SCRATCHBIG/dataset
modeldir=$SCRATCHBIG/microsoft/codebert-base
syntheticdataset=../synthetic_data/d2c_semisynthetic.jsonl


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
    --synthetic_data_file=$syntheticdataset \
    --num_train_epochs $((n_partitions*3)) \
    --num_train_examples $n_examples \
    --num_synthetic_examples 0 \
    --synthetic_example_offset $((partition*n_examples)) \
    --train_example_offset $((partition*n_examples)) \
    --block_size 256 \
    --train_batch_size $batchsize \
    --eval_batch_size 64 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --gradient_checkpointing \
    --save_steps $((3200 / batch_size)) \
    --early_stopping_patience 32 \
    --seed $seed 2>&1| tee train.log


mkdir -p ./saved_models/$model_path
rm -rf ./saved_models/$model_path
cp -r $savedir ./saved_models/$model_path
