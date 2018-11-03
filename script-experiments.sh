#!/usr/bin/env bash

set -e

export KERAS_BACKEND=theano

FACTORS=( 8 16 32 64 )
INPUT_DIR=$1
DATASET=$2
EPOCHS=$3
WTE=$4 # what to evaluate

echo "running experiments with $EPOCHS epochs"

if [ $WTE = "factors" ]
then
    echo "evaluating factors..."
    # fix all, evaluate factors
    for f in "${FACTORS[@]}";
    do
        echo "factors "$f
        python GMF.py --path $INPUT_DIR --dataset $DATASET --epochs $EPOCHS --num_factors $f
        python MLP.py --path $INPUT_DIR --dataset $DATASET --epochs $EPOCHS --num_factors $f
        python NeuMF.py --path $INPUT_DIR --dataset $DATASET --epochs $EPOCHS --num_factors $f
    done;
fi

if [ $WTE = "negatives" ]
then
    NEG=10
    for n in $(seq 1 $NEG);
    do
        echo "negatives: "$n
        python GMF.py --path $INPUT_DIR --dataset $DATASET --epochs $EPOCHS --num_neg $n --out 0
        python MLP.py --path $INPUT_DIR --dataset $DATASET --epochs $EPOCHS --num_neg $n --out 0
        python NeuMF.py --path $INPUT_DIR --dataset $DATASET --epochs $EPOCHS --num_neg $n --out 0
    done;
fi

if [ $WTE = "topk" ]
then
    TOPK=10
    for k in $(seq 1 $TOPK);
    do
        echo "negatives: "$n
        python GMF.py --path $INPUT_DIR --dataset $DATASET --epochs $EPOCHS --topk $k --out 0
        python MLP.py --path $INPUT_DIR --dataset $DATASET --epochs $EPOCHS --topk $k --out 0
        python NeuMF.py --path $INPUT_DIR --dataset $DATASET --epochs $EPOCHS --topk $k --out 0
    done;
fi

