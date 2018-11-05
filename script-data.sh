#!/usr/bin/env bash

set -e

mkdir input
mkdir pretrain
mkdir results


wget -O input/ml-100k.zip https://storage.googleapis.com/ml-research-datasets/recsys/ml-100k-v2.zip
unzip input/ml-100k.zip -d input/

wget -O input/ml-1m.zip https://storage.googleapis.com/ml-research-datasets/recsys/ml-1m-v2.zip
unzip input/ml-1m.zip -d input/

