#! /bin/bash
eval "$(micromamba shell hook --shell=bash)"

lang=$1
#savedir=$SCRATCHBIG/saved_models/$lang
savedir=~/codebert-base-trained/$lang
export TRANSFORMERS_OFFLINE=1

datasetdir=$SCRATCHBIG/dataset

if [[ ! -d $datasetdir ]]
then 
    bash copy_to_scratch.sh
fi

micromamba activate ML3.8

cd code
python ../evaluator/evaluator.py \
    -a $datasetdir/test.jsonl \
    -p $savedir/predictions.jsonl
