#!/bin/bash -e
#
# Usage: ./run.sh <part#>
# Example: ./run.sh 1
#

if [ $1 == 1 ] ; then
    echo "running train.py"
    python train.py --batch_size 100 --epochs 40 --gpuid 1
    # python train.py --data_file data/hw4_data.bin --optimizer Adam -lr 1e-2 --batch_size 48 --model_file model.py
elif [ $1 == 2 ] ; then
    echo "running train_bi.py"
    python train_bi.py --data_file data/hw4_data.bin --optimizer Adam -lr 1e-2 --batch_size 48 --model_file model.py
elif [ $1 == 3 ] ; then
    echo "running train_dropout.py"
    python train_dropout.py --data_file data/hw4_data.bin --optimizer Adam -lr 1e-2 --batch_size 48 --model_file model.py
elif [ $1 == 4 ] ; then
    echo "running train_cuda_dropout.py"
    python train_cuda_dropout.py --data_file data/hw4_data.bin --optimizer Adam -lr 1e-2 --batch_size 48 --model_file model.py --gpuid 1
else
    echo "Unknown parameter"
fi
