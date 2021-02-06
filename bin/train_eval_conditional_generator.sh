#!/bin/bash
set -e

PY_PATH='~/anaconda3/bin/python'
TRAIN_FROM_SCRATCH='Y'
AUTO_EVAL='Y'
DATA_DIR='./data/processed/wiki2016_gpt2_small/'

if [ $TRAIN_FROM_SCRATCH == "Y" ]; then
    OUPUT_PATH_PREFIX='./models/conditional_all'
    eval $PY_PATH src/main_train_condition.py --save $OUPUT_PATH_PREFIX --data $DATA_DIR --tensor_folder tensors_all_min100 --batch_size 7 --bptt 256 --num_insert 5 --n_further 25 --training_split_num 5 --valid_per_epoch 5 --turn_off_condition False --optimizer AdamW --lr 1e-4 --wdecay 0
else
    LAST_MODEL_DIR='./models/conditional_all-20200106-235956'
    eval $PY_PATH src/main_train_condition.py --save $LAST_MODEL_DIR --data $DATA_DIR --tensor_folder tensors_all_min100 --batch_size 7 --bptt 256 --num_insert 5 --n_further 25 --training_split_num 5 --valid_per_epoch 5 --turn_off_condition False --optimizer AdamW --lr 1e-4 --wdecay 0 --continue_train --start_training_split 0
fi

if [ $AUTO_EVAL == "Y" ]; then
    OPTION_MODEL_PATH=
    eval $PY_PATH src/condition_LM_test.py --checkpoint_org_for_pplm ./models/conditional_all-20200106-235956 --checkpoint_topics ./models/future_topic_all-20200106-222318_3w --checkpoint_conditional ./models/conditional_all-20200106-235956 --batch_size 1 --bptt 512 --bptt_conditional 256 --dilated_head_span 80 --outf gen_log/conditional_wiki2016_all_n10_NSD_no_filter_PPLM_fine_tuned_cond_fix_len_large_bz1_BLEU_fix --tensor_folder tensors_all_min100 --data ./data/processed/wiki2016_gpt2/ --n_basis 10 --de_en_connection False --max_batch_num 300 --seed 0
fi

