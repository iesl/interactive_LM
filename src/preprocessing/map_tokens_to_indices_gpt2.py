import argparse
import gzip
import os
import time
import sys
sys.path.insert(0, sys.path[0]+'/..')
from utils import Dictionary, str2bool

#map words to index (and set the max sentence number), 
#map low freq words into <unk>
#add end of sentence tokens (for transformer to generate output embedding), 
#output dataset, dictionary (start with <null>, <unk>, and <eos>), word frequency,  print total number of words, total number of filtered words

parser = argparse.ArgumentParser(description='Preprocessing step 1')
#parser.add_argument('--data', type=str, default='./data/raw/binary-wackypedia-1-4-ukwac-.gz',
parser.add_argument('--data', type=str, default='./data/raw/wiki2016_gpt2.txt',
                    help='location of the data corpus')
#/iesl/canvas/xiangl/data/bookcorpus/books_large_p1.txt
#parser.add_argument('--save', type=str, default='./data/processed/wackypedia/',
parser.add_argument('--save', type=str, default='./data/processed/wiki2016_gpt2/',
                    help='path to save the output data')
#parser.add_argument('--min_freq', type=int, default='5',
#                    help='map to <unk> if observe less than this number')
#parser.add_argument('--min_sent_length', type=int, default='5',
#                    help='skip the sentence if sentence length is less than this number')
parser.add_argument('--max_sent_num', type=int, default='100000000000000',
#parser.add_argument('--max_sent_num', type=int, default='100',
                    help='load only this number of sentences from input corpus')
#parser.add_argument('--lowercase', type=str2bool, nargs='?', default=False,
#                    help='whether make all the words in corpus lowercased')

args = parser.parse_args()

print(args)

start_time = time.time()

corpus_output_name = args.save + "corpus_index"
dictionary_output_name = args.save + "dictionary_index"

if not os.path.exists(args.save):
    os.makedirs(args.save)

f_out = open(corpus_output_name, 'w')

my_open = open
byte_mode = False
        
dict_c = Dictionary(byte_mode)

total_num_w = 0
with my_open(args.data, 'r') as f_in:
    for line_idx, line in enumerate(f_in):
        sent_spacy, gpt2_idx, gpt2_mapping = line.rstrip().split('\t')
        w_ind_list = []
        for w in sent_spacy.split():
            w_ind = dict_c.dict_check_add(w)
            w_ind_list.append(w_ind)
            total_num_w += 1
        dict_c.append_eos(w_ind_list)
        f_out.write(' '.join([str(x) for x in w_ind_list]) + '\t' + gpt2_idx + '\t' + gpt2_mapping + '\n')
        if line_idx % 1000000 == 0:
            print(line_idx)
            sys.stdout.flush()
        if line_idx >= args.max_sent_num:
            break

f_out.close()

with open(dictionary_output_name, 'w') as f_out:
    dict_c.store_dict(f_out)


elapsed = time.time() - start_time
print("time of total word to index: "+str(elapsed)+'s')
