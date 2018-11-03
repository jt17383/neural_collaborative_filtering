#!/usr/bin/env bash

export KERAS_BACKEND=theano

FACTORS=( 8 16 32 64 )
INPUT_DIR=$1
DATASET=$2
EPOCHS=$3

echo "running experiments with $EPOCHS epochs"

# fix negatives, evaluate factors

for f in "${FACTORS[@]}";
do
    echo "factors "$f
    python GMF.py --path $INPUT_DIR --dataset $DATASET --epochs $EPOCHS --num_factors $f
    python MLP.py --path $INPUT_DIR --dataset $DATASET --epochs $EPOCHS --num_factors $f
    python NeuMF.py --path $INPUT_DIR --dataset $DATASET --epochs $EPOCHS --num_factors $f
done;

# fix factors, evaluate negatives
NEG=10
for n in $(seq 1 $NEG);
do
    echo "negatives: "$n
    python GMF.py --path $INPUT_DIR --dataset $DATASET --epochs $EPOCHS --num_neg $n
    python MLP.py --path $INPUT_DIR --dataset $DATASET --epochs $EPOCHS --num_neg $n
    python NeuMF.py --path $INPUT_DIR --dataset $DATASET --epochs $EPOCHS --num_neg $n
done;


# fix factors, negatives; evaluate k
TOPK=10
for k in $(seq 1 $TOPK);
do
    echo "negatives: "$n
    python GMF.py --path $INPUT_DIR --dataset $DATASET --epochs $EPOCHS --topk $k
    python MLP.py --path $INPUT_DIR --dataset $DATASET --epochs $EPOCHS --topk $k
    python NeuMF.py --path $INPUT_DIR --dataset $DATASET --epochs $EPOCHS --topk $k
done;
