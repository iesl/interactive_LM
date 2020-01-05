#!/bin/bash

#SBATCH --ntasks=5
#sbatch --ntasks=50 --ntasks-per-node=10 --nodes=5 --cpus-per-task=1 --mem-per-cpu=10G ./bin/tokenize_all_wiki_gpt2.sh

#sacct -j 89262 --format=JobID,Start,End,Elapsed,NCPUS,nodelist,JobName
#INPUT_DIR="/iesl/data/nmonath-wiki-json/json/"
#OUTPUT_DIR="/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/data/raw/wiki2016_both/"
INPUT_DIR="/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/data/raw/wiki2016_both/"
#OUTPUT_DIR="/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/data/raw/wiki2016_gpt2/"
OUTPUT_DIR="/iesl/canvas/hschang/language_modeling/interactive_LM/data/raw/wiki2016_gpt2_order/"


for file_path in $INPUT_DIR/*; do
    file_name=`basename $file_path`
    output_file_name="${file_name}"
    echo --ntasks=1 --nodes=1 --exclusive -p cpu ~/anaconda3/bin/python src/preprocessing/tools/tokenize_gpt2.py -i $file_path -o $OUTPUT_DIR/$output_file_name
    srun --ntasks=1 --nodes=1 --exclusive -p cpu ~/anaconda3/bin/python src/preprocessing/tools/tokenize_gpt2.py -i $file_path -o $OUTPUT_DIR/$output_file_name  &
    #srun --ntasks=1 --nodes=1 --mem=100G -p cpu ~/anaconda3/bin/python src/preprocessing/tokenize_wiki.py -i $file_path -o $OUTPUT_DIR/$output_file_name
    #srun --ntasks=1 --mem=100G -p cpu ~/anaconda3/bin/python src/preprocessing/tokenize_wiki.py -i $file_path -o $OUTPUT_DIR/$output_file_name
    #sleep 2
done
wait
