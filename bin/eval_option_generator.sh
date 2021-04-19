#!/bin/bash
set -e

#PY_PATH='~/anaconda3/bin/python'
PY_PATH='~/anaconda3/envs/interactive/bin/python'
VISUALIZATION='Y'
AUTO_EVAL_TOPICS='Y' 
AUTO_EVAL_GENRATION='Y' #require the conditional generation model
OUTPUT_CSV_FOR_MTURK='N' #require the conditional generation model
TOPIC_MODEL_DIR='./models/future_topic_all-20200106-222318'
GENERATION_MODEL_DIR='./models/conditional_all-20200106-235956'
DATA_DIR='./data/processed/wiki2016_gpt2/'
STOP_WORD_FILE='./resources/stop_word_list'
OUPUT_FILE_DIR='./gen_log'

LDA_MODEL_PATH='./models/LDA/LDA_wiki2016_150' #optional
WORD_CLUSTER_PATH='./models/GloVe_clustering/normalized_center_emb_n1000' #optional

mkdir -p $OUPUT_FILE_DIR

if [ $VISUALIZATION == "Y" ]; then
    OUT_FILE="$OUPUT_FILE_DIR/vis_topics_wiki2016"
    eval $PY_PATH src/evaluation/basis_test.py --checkpoint_topics $TOPIC_MODEL_DIR  --outf $OUT_FILE --data $DATA_DIR --topic_models NSD_vis --batch_size 5 --bptt 512 --dilated_head_span 200 --tensor_folder tensors_all_min100 --n_basis 10 --de_en_connection False
fi

if [ $AUTO_EVAL_TOPICS == "Y" ]; then
    OUT_FILE="$OUPUT_FILE_DIR/topics_wiki2016" #scores are stored at the end of the file
    eval $PY_PATH src/evaluation/basis_test.py --topic_models NSD+kmeans_cluster+SC_cluster+LDA_org+random_word+random_vocab+global_centers --checkpoint_topics $TOPIC_MODEL_DIR --LDA_model_path $LDA_MODEL_PATH --word_emb_center_path $WORD_CLUSTER_PATH --outf $OUT_FILE --data $DATA_DIR --stop_word_file $STOP_WORD_FILE --batch_size 3 --bptt 512 --dilated_head_span 80 --tensor_folder tensors_all_min100 --n_basis 10 --de_en_connection False --max_batch_num 100
fi

if [ $AUTO_EVAL_GENRATION == "Y" ]; then
    METHOD_LIST="NSD_topic kmeans_cluster SC_cluster random_word random_vocab no_condition global_centers LDA_org "

    # Iterate the string variable using for loop
    for OPTION_METHOD in $METHOD_LIST; do
        echo "Evaluating method $OPTION_METHOD"
        OUT_FILE="$OUPUT_FILE_DIR/topics_generation_wiki2016_$OPTION_METHOD"
        eval $PY_PATH src/evaluation/condition_LM_test.py --topic_mode $OPTION_METHOD --checkpoint_topics $TOPIC_MODEL_DIR --checkpoint_conditional $GENERATION_MODEL_DIR --outf $OUT_FILE --LDA_model_path $LDA_MODEL_PATH --word_emb_center_path $WORD_CLUSTER_PATH --stop_word_file $STOP_WORD_FILE --batch_size 1 --bptt 512 --bptt_conditional 256 --dilated_head_span 80 --tensor_folder tensors_all_min100 --data $DATA_DIR --n_basis 10 --de_en_connection False --max_batch_num 300 --seed 0 #Notice that our dist-n metrics depend on the setting of batch size

    done
fi

if [ $OUTPUT_CSV_FOR_MTURK == "Y" ]; then
    STS_FILE='./data/raw/stsbenchmark/sts-train_longer_uniq.csv'
    METHOD_LIST='NSD_topic kmeans_cluster global_centers LDA_org'
    OUTPUT_CSV="$OUPUT_FILE_DIR/output_STSb_LDA_lkmeans_gkmeans_NSD_example.csv"

    # Iterate the string variable using for loop
    for OPTION_METHOD in $METHOD_LIST; do
        echo "Evaluating method $OPTION_METHOD"
        OUT_FILE="$OUPUT_FILE_DIR/topics_generation_STSb_$OPTION_METHOD"
        OUT_CSV_FILE="$OUPUT_FILE_DIR/topics_generation_STSb_${OPTION_METHOD}.csv"
        eval $PY_PATH src/evaluation/condition_LM_test.py --topic_mode $OPTION_METHOD --checkpoint_topics $TOPIC_MODEL_DIR --checkpoint_conditional $GENERATION_MODEL_DIR --STS_location $STS_FILE --outf $OUT_CSV_FILE --csv_outf $OUT_CSV_FILE --data $DATA_DIR --LDA_model_path $LDA_MODEL_PATH --word_emb_center_path $WORD_CLUSTER_PATH --stop_word_file $STOP_WORD_FILE --use_corpus STS --batch_size 3 --bptt 512 --bptt_conditional 256 --dilated_head_span 80 --readable_context True --tensor_folder tensors_all_min100 --n_basis 10 --de_en_connection False --max_batch_num 50 --seed 0 --num_sent_gen 3
    done
    eval $PY_PATH src/evaluation/concate_task23_csv.py -p "$OUPUT_FILE_DIR/topics_generation_STSb_" -o $OUTPUT_CSV -m "${METHOD_LIST// /\|}"
fi
