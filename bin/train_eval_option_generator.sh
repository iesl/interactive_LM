#!/bin/bash
set -e

PY_PATH='~/anaconda3/bin/python'
TRAIN_FROM_SCRATCH='Y'
VISUALIZATION='Y'
AUTO_EVAL_TOPICS='Y' 
AUTO_EVAL_GENRATION='Y' #require the conditional generation model
DATA_DIR='./data/processed/wiki2016_gpt2_small/'
EMB_FILE='/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/resources/glove.840B.300d_filtered_wiki2016_min100.txt'

if [ $TRAIN_FROM_SCRATCH == "Y" ]; then
    OUPUT_PATH_PREFIX='./models/future_topic_all'
    eval $PY_PATH src/main_train_topics.py --save $OUPUT_PATH_PREFIX --data $DATA_DIR --emb_file $EMB_FILE --tensor_folder tensors_all_min100 --coeff_opt lc --batch_size 30 --bptt 512 --dilated_head_span 200 --de_model TRANS --trans_layers 5 --n_basis 10 --n_further 50 --training_split_num 2 --valid_per_epoch 2 --de_en_connection False
else
    LAST_MODEL_DIR='./models/future_topic_all-20200106-222318_3w'
    eval $PY_PATH src/main_train_topics.py --save $LAST_MODEL_DIR --data $DATA_DIR --emb_file $EMB_FILE --tensor_folder tensors_all_min100 --coeff_opt lc --batch_size 30 --bptt 512 --dilated_head_span 200 --de_model TRANS --trans_layers 5 --n_basis 10 --n_further 50 --training_split_num 2 --valid_per_epoch 2 --de_en_connection False --continue_train --start_training_split 0
fi


if [ $VISUALIZATION == "Y" ]; then
fi

if [ $AUTO_EVAL == "Y" ]; then
    eval $PY_PATH src/basis_test.py --topic_models NSD+kmeans_cluster+SC_cluster+LDA_org+random_word+random_vocab+global_centers --checkpoint_topics ./models/future_topic_all-20200106-222318_3w --batch_size 3 --bptt 512 --dilated_head_span 80 --outf gen_log/wiki2016_all_glove_trans_bsz10_n10_all_test_large_final_3w_f50 --data ./data/processed/wiki2016_gpt2/ --tensor_folder tensors_all_min100 --n_basis 10 --de_en_connection False --LDA_model_path /iesl/canvas/hschang/language_modeling/interactive_LM/models/LDA/LDA_wiki2016_150 --word_emb_center_path /iesl/canvas/hschang/language_modeling/interactive_LM/models/GloVe_clustering/normalized_center_emb_n1000 --max_batch_num 100 #--cuda False
fi

if [ $AUTO_EVAL_GENRATION == "Y" ]; then
    GENERATION_MODEL=''
    #eval $PY_PATH src/condition_LM_test.py  --checkpoint_topics ./models/future_topic_all-20200106-222318_3w --checkpoint_conditional ./models/conditional_all-20200106-235956 --batch_size 1 --bptt 512 --bptt_conditional 256 --dilated_head_span 80 --outf gen_log/conditional_wiki2016_all_NSD_3w_topic_ppl_medium_large_bz1_BLEU_fix --tensor_folder tensors_all_min100 --data ./data/processed/wiki2016_gpt2/ --n_basis 10 --de_en_connection False --max_batch_num 300 --seed 0 --topic_mode NSD_topic
#eval $PY_PATH src/condition_LM_test.py  --checkpoint_topics ./models/future_topic_all-20200106-222318_3w --checkpoint_conditional ./models/conditional_all-20200106-235956 --batch_size 1 --bptt 512 --bptt_conditional 256 --dilated_head_span 80 --outf gen_log/conditional_wiki2016_all_kmeans_cluster_3w_topic_ppl_medium_large_bz1_BLEU_fix --tensor_folder tensors_all_min100 --data ./data/processed/wiki2016_gpt2/ --n_basis 10 --de_en_connection False --max_batch_num 300 --seed 0 --topic_mode kmeans_cluster
#eval $PY_PATH src/condition_LM_test.py  --checkpoint_topics ./models/future_topic_all-20200106-222318_3w --checkpoint_conditional ./models/conditional_all-20200106-235956 --batch_size 1 --bptt 512 --bptt_conditional 256 --dilated_head_span 80 --outf gen_log/conditional_wiki2016_all_random_word_topic_large_bz1_BLEU_fix --tensor_folder tensors_all_min100 --data ./data/processed/wiki2016_gpt2/ --n_basis 10 --de_en_connection False --max_batch_num 300 --seed 0 --topic_mode random_word
#eval $PY_PATH src/condition_LM_test.py  --checkpoint_topics ./models/future_topic_all-20200106-222318_3w --checkpoint_conditional ./models/conditional_all-20200106-235956 --batch_size 1 --bptt 512 --bptt_conditional 256 --dilated_head_span 80 --outf gen_log/conditional_wiki2016_all_LDA_org_topic_large_bz1_BLEU_fix --tensor_folder tensors_all_min100 --data ./data/processed/wiki2016_gpt2/ --n_basis 10 --de_en_connection False --max_batch_num 300 --seed 0 --topic_mode LDA_org --LDA_model_path /iesl/canvas/hschang/language_modeling/interactive_LM/models/LDA/LDA_wiki2016_150
#eval $PY_PATH src/condition_LM_test.py  --checkpoint_topics ./models/future_topic_all-20200106-222318_3w --checkpoint_conditional ./models/conditional_all-20200106-235956 --batch_size 1 --bptt 512 --bptt_conditional 256 --dilated_head_span 80 --outf gen_log/conditional_wiki2016_all_SC_cluster_topic_iter1000_large_shuffle_bz1_BLEU_fix --tensor_folder tensors_all_min100 --data ./data/processed/wiki2016_gpt2/ --n_basis 10 --de_en_connection False --max_batch_num 300 --seed 0 --topic_mode SC_cluster
#eval $PY_PATH src/condition_LM_test.py  --checkpoint_topics ./models/future_topic_all-20200106-222318_3w --checkpoint_conditional ./models/conditional_all-20200106-235956 --batch_size 1 --bptt 512 --bptt_conditional 256 --dilated_head_span 80 --outf gen_log/conditional_wiki2016_all_global_clusters_topic_large_fixed_norm_bz1_BLEU_fix --tensor_folder tensors_all_min100 --data ./data/processed/wiki2016_gpt2/ --n_basis 10 --de_en_connection False --max_batch_num 300 --seed 0 --topic_mode global_centers --word_emb_center_path /iesl/canvas/hschang/language_modeling/interactive_LM/models/GloVe_clustering/normalized_center_emb_n1000
#eval $PY_PATH src/condition_LM_test.py  --checkpoint_topics ./models/future_topic_all-20200106-222318_3w --checkpoint_conditional ./models/conditional_all-20200106-235956 --batch_size 1 --bptt 512 --bptt_conditional 256 --dilated_head_span 80 --outf gen_log/conditional_wiki2016_all_random_vocab_topic_large_bz1_BLEU_fix --tensor_folder tensors_all_min100 --data ./data/processed/wiki2016_gpt2/ --n_basis 10 --de_en_connection False --max_batch_num 300 --seed 0 --topic_mode random_vocab
#eval $PY_PATH src/condition_LM_test.py  --checkpoint_topics ./models/future_topic_all-20200106-222318_3w --checkpoint_conditional ./models/conditional_all-20200106-235956 --batch_size 1 --bptt 512 --bptt_conditional 256 --dilated_head_span 80 --outf gen_log/conditional_wiki2016_all_no_topic_ppl_medium_large_bz1_BLEU_fix --tensor_folder tensors_all_min100 --data ./data/processed/wiki2016_gpt2/ --n_basis 10 --de_en_connection False --max_batch_num 300 --seed 0 --topic_mode no_condition
fi
