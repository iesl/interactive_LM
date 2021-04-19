#!/bin/bash
set -e
echo "code is running"

PY_PATH="~/anaconda3/bin/python"
#PY_PATH="~/anaconda3/envs/interactive/bin/python"

INPUT_JSON_DIR="./data/raw/wikipedia/"
STOP_WORD_LIST="./resources/stop_word_list"

OUTPUT_SPACY_DIR="./data/raw/wiki2016_both/"
OUTPUT_GPT2_DIR="./data/raw/wiki2016_gpt2_order/"
OUTPUT_GPT2="./data/raw/wiki2016_gpt2.txt"
OUTPUT_GPT2_PROC_DIR="./data/processed/wiki2016_gpt2/"
OUTPUT_GPT2_TENSOR_DIR="$OUTPUT_GPT2_PROC_DIR/tensors_all_min100/"
OUTPUT_DICT_FILE="$OUTPUT_GPT2_TENSOR_DIR/dict_idx_compact"

mkdir -p $OUTPUT_SPACY_DIR
mkdir -p $OUTPUT_GPT2_DIR
mkdir -p $OUTPUT_GPT2_PROC_DIR

TOKENIZE_MODE=2

for file_path in $INPUT_JSON_DIR/*; do
    file_name=`basename $file_path`
    output_file_name="${file_name//.json.gz/}"
    
    eval $PY_PATH src/preprocessing/tools/tokenize_wiki.py -i $file_path -t $TOKENIZE_MODE -o $OUTPUT_SPACY_DIR/$output_file_name #Convert json files to plain text and tokenize the text using Spacy
    eval $PY_PATH src/preprocessing/tools/tokenize_gpt2.py -i $OUTPUT_SPACY_DIR/$output_file_name -o $OUTPUT_GPT2_DIR/$output_file_name #Tokenize the text for GPT2
done

cat $OUTPUT_GPT2_DIR/* > $OUTPUT_GPT2
eval $PY_PATH src/preprocessing/map_tokens_to_indices_gpt2.py --data $OUTPUT_GPT2 --save $OUTPUT_GPT2_PROC_DIR #Convert the output of Spacy tokenizer to word index
eval $PY_PATH src/preprocessing/map_indices_to_tensors_gpt2_memory_saving.py --data $OUTPUT_GPT2_PROC_DIR --save $OUTPUT_GPT2_TENSOR_DIR --new_dict_file $OUTPUT_DICT_FILE --stop_word_file $STOP_WORD_LIST #Align tokens from Spacy and from GPT2
