#!/bin/bash
set -e

PY_PATH='~/anaconda3/bin/python'
#PY_PATH='~/anaconda3/envs/interactive/bin/python'
AUTO_EVAL='Y'
OUTPUT_CSV_FOR_MTURK='N'

GENERATION_MODEL_DIR='./models/conditional_all-20200106-235956'
TOPIC_MODEL_DIR='./models/future_topic_all-20200106-222318'
STOP_WORD_FILE='./resources/stop_word_list'

DATA_DIR='./data/processed/wiki2016_gpt2/'
OUPUT_FILE_DIR='./gen_log'

mkdir -p $OUPUT_FILE_DIR

if [ $AUTO_EVAL == "Y" ]; then
    OUT_FILE="$OUPUT_FILE_DIR/conditional_generation_wiki2016"
    eval $PY_PATH src/evaluation/condition_LM_test.py --checkpoint_org_for_pplm $GENERATION_MODEL_DIR --checkpoint_topics $TOPIC_MODEL_DIR --checkpoint_conditional $GENERATION_MODEL_DIR --data $DATA_DIR --outf $OUT_FILE --stop_word_file $STOP_WORD_FILE --topic_mode NSD --batch_size 1 --bptt 512 --bptt_conditional 256 --dilated_head_span 80 --tensor_folder tensors_all_min100 --n_basis 10 --de_en_connection False --max_batch_num 300 --seed 0 #Notice that our dist-n metrics depend on the setting of batch size
fi

if [ $OUTPUT_CSV_FOR_MTURK == "Y" ]; then
    STS_FILE='./data/raw/stsbenchmark/sts-train_longer_uniq.csv'
    OUT_FILE="$OUPUT_FILE_DIR/conditional_generation_STSb"
    OUT_CSV_FILE="$OUPUT_FILE_DIR/conditional_generation_STSb.csv"
    eval $PY_PATH src/evaluation/condition_LM_test.py --checkpoint_org_for_pplm $GENERATION_MODEL_DIR --checkpoint_topics $TOPIC_MODEL_DIR --checkpoint_conditional $GENERATION_MODEL_DIR --outf $OUT_FILE --csv_outf $OUT_CSV_FILE --STS_location $STS_FILE --data $DATA_DIR --stop_word_file $STOP_WORD_FILE --use_corpus STS --topic_mode NSD --readable_context True --batch_size 3 --bptt 512 --bptt_conditional 256 --dilated_head_span 80 --tensor_folder tensors_all_min100 --n_basis 10 --de_en_connection False --max_batch_num 50 --seed 0 --num_sent_gen 3
fi
