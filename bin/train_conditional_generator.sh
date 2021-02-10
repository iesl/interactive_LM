#!/bin/bash
set -e

PY_PATH='~/anaconda3/bin/python'
#PY_PATH='~/anaconda3/envs/interactive/bin/python'
TRAIN_FROM_SCRATCH='Y'
DATA_DIR='./data/processed/wiki2016_gpt2/'
EMB_FILE='./resources/glove.840B.300d.txt'

if [ $TRAIN_FROM_SCRATCH == "Y" ]; then
    OUPUT_PATH_PREFIX='./models/conditional_all'
    eval $PY_PATH src/main_train_condition.py --save $OUPUT_PATH_PREFIX --data $DATA_DIR --emb_file $EMB_FILE --tensor_folder tensors_all_min100 --batch_size 7 --bptt 256 --num_insert 5 --n_further 25 --training_split_num 5 --valid_per_epoch 5 --turn_off_condition False --optimizer AdamW --lr 1e-4 --wdecay 0
else
    LAST_MODEL_DIR='./models/conditional_all-20200106-235956'
    eval $PY_PATH src/main_train_condition.py --save $LAST_MODEL_DIR --data $DATA_DIR --emb_file $EMB_FILE --tensor_folder tensors_all_min100 --batch_size 7 --bptt 256 --num_insert 5 --n_further 25 --training_split_num 5 --valid_per_epoch 5 --turn_off_condition False --optimizer AdamW --lr 1e-4 --wdecay 0 --continue_train --start_training_split 0
fi

