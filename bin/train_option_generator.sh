#!/bin/bash
set -e

PY_PATH='~/anaconda3/bin/python'
#PY_PATH="~/anaconda3/envs/interactive/bin/python"
TRAIN_FROM_SCRATCH='Y'
DATA_DIR='./data/processed/wiki2016_gpt2/'
#DATA_DIR='./data/processed/wiki2016_gpt2/'
EMB_FILE='./resources/glove.840B.300d.txt'

if [ $TRAIN_FROM_SCRATCH == "Y" ]; then
    OUPUT_PATH_PREFIX='./models/future_topic_all'
    eval $PY_PATH src/main_train_topics.py --save $OUPUT_PATH_PREFIX --data $DATA_DIR --emb_file $EMB_FILE --tensor_folder tensors_all_min100 --coeff_opt lc --batch_size 30 --bptt 512 --dilated_head_span 200 --de_model TRANS --trans_layers 5 --n_basis 10 --n_further 50 --training_split_num 2 --valid_per_epoch 2 --de_en_connection False
else
    LAST_MODEL_DIR='./models/future_topic_all-20200106-222318'
    eval $PY_PATH src/main_train_topics.py --save $LAST_MODEL_DIR --data $DATA_DIR --emb_file $EMB_FILE --tensor_folder tensors_all_min100 --coeff_opt lc --batch_size 30 --bptt 512 --dilated_head_span 200 --de_model TRANS --trans_layers 5 --n_basis 10 --n_further 50 --training_split_num 2 --valid_per_epoch 2 --de_en_connection False --continue_train --start_training_split 0
fi


