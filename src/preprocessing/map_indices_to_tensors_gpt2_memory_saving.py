import argparse
import torch
import sys
import random
import os
sys.path.insert(0, sys.path[0]+'/..')
#sys.path.append("..")
import utils
import numpy as np

#remove the duplicated sentences
#remove the sentences which are too long
#remove stop words in target
#handle the min target filtering (If more than 30 words in output, just do random sampling)
#padding and store them into tensors, (random shuffle? two sets), store train, val, and test

parser = argparse.ArgumentParser(description='Preprocessing step 2 for GPT2')
parser.add_argument('--data', type=str, default='./data/processed/wiki2016_gpt2/',
                    help='location of the data corpus')
#parser.add_argument('--save', type=str, default='./data/processed/wiki2016_gpt2/tensors/',
#parser.add_argument('--save', type=str, default='./data/processed/wiki2016_gpt2/tensors_100000_min100_test/',
#parser.add_argument('--save', type=str, default='./data/processed/wiki2016_gpt2/tensors_10000000_min100/',
#parser.add_argument('--save', type=str, default='./data/processed/wiki2016_gpt2/tensors_70000000_min100/',
parser.add_argument('--save', type=str, default='./data/processed/wiki2016_gpt2/tensors_all_min100/',
#parser.add_argument('--save', type=str, default='./data/processed/wackypedia/tensors_multi150/',
                    help='path to save the output data')
#parser.add_argument('--new_dict_file', type=str, default='./data/processed/wiki2016_gpt2/tensors_100000_min100_test/dict_idx_compact',
#parser.add_argument('--new_dict_file', type=str, default='./data/processed/wiki2016_gpt2/tensors_10000000_min100/dict_idx_compact',
#parser.add_argument('--new_dict_file', type=str, default='./data/processed/wiki2016_gpt2/tensors_70000000_min100/dict_idx_compact',
parser.add_argument('--new_dict_file', type=str, default='./data/processed/wiki2016_gpt2/tensors_all_min100/dict_idx_compact',
                    help='path to save the output data')
#parser.add_argument('--max_sent_len', type=int, default=50,
#parser.add_argument('--max_sent_len', type=int, default=150,
#                    help='max sentence length for input features')
#parser.add_argument('--multi_sent', default=False, action='store_true',
#parser.add_argument('--multi_sent', default=False, 
#                    help='Whether do we want to cram multiple sentences into one input feature')
#parser.add_argument('--max_target_num', type=int, default=30,
#                    help='max word number for output prediction w/o stop words (including above and below sentences)')
parser.add_argument('--min_freq', type=int, default='100',
                    help='map to <unk> if observe less than this number')
#parser.add_argument('--max_sent_num', type=int, default='100000',
#parser.add_argument('--max_sent_num', type=int, default='10000000',
#parser.add_argument('--max_sent_num', type=int, default='70000000',
parser.add_argument('--max_sent_num', type=int, default='10000000000',
                    help='load only this number of sentences from input corpus')
parser.add_argument('--testing_ratio', type=float, default='0.05',
                    help='the portion of testing data, which is the same as the portion of validation data')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--stop_word_file', type=str, default='./resources/stop_word_list',
                    help='path to the file of a stop word list')

EOS_IDX = 2

args = parser.parse_args()

print(args)

random.seed(args.seed)

if not os.path.exists(args.save):
    os.makedirs(args.save)

#def convert_stop_to_ind(f_in, w_d2_ind_freq):
#    stop_word_set = set()
#    for line in f_in:
#        w = line.rstrip()
#        if w in w_d2_ind_freq:
#            stop_word_set.add(w_d2_ind_freq[w][0])
#    return stop_word_set

def convert_stop_to_ind_lower(f_in, idx2word_freq):
    stop_word_org_set = set()
    for line in f_in:
        w = line.rstrip()
        stop_word_org_set.add(w)
    stop_word_set = set()
    for idx, (w, freq) in enumerate(idx2word_freq):
        if w.lower() in stop_word_org_set:
            stop_word_set.add(idx)
    return stop_word_set
        
def densify_index(w_ind_spacy_corpus, idx2word_freq):
    w_ind_spacy_corpus_new = []
    num_special_tok = 3
    idx2word_freq_new = [[w, freq] for w, freq in idx2word_freq[:num_special_tok]]
    existing_idx_dict = {x: x for x in range(num_special_tok)}
    for w_idx in w_ind_spacy_corpus:
        if w_idx in existing_idx_dict:
            w_idx_compact = existing_idx_dict[w_idx]
        else:
            w_idx_compact = len(existing_idx_dict)
            existing_idx_dict[w_idx]= w_idx_compact
            idx2word_freq_new.append(idx2word_freq[w_idx])
        w_ind_spacy_corpus_new.append(w_idx_compact)
    return w_ind_spacy_corpus_new, idx2word_freq_new

#def load_w_ind(f_in, max_sent_num, max_sent_len):
def load_w_ind(f_in, max_sent_num, idx2word_freq, min_freq, stop_ind_set, total_num_words, total_num_words_spacy):
    w_ind_gpt2_corpus = np.zeros(total_num_words, dtype='int32')
    w_ind_spacy_corpus = np.zeros(total_num_words_spacy, dtype='int32')
    idx_gpt2_to_spacy = np.zeros(total_num_words, dtype='int32')
    #last_sent = ''
    #num_duplicated_sent = 0
    #num_too_long_sent = 0
    num_stop_words = 0
    num_rare_words = 0
    
    current_num_word_gpt2 = 0
    current_num_word_spacy = 0
    for line_idx, line in enumerate(f_in):
        sent_spacy, sent_gpt2, gpt2_to_spacy = line.rstrip().split('\t')
        gpt2_idx = [int(x) for x in sent_gpt2.split()]
        w_ind_gpt2_corpus[current_num_word_gpt2:current_num_word_gpt2+len(gpt2_idx)] = gpt2_idx
        org_idx_l2_compact_idx = []
        current_num_word_spacy_old = current_num_word_spacy
        #valid_w_idx = []
        for w_idx_str in sent_spacy.split():
            w_idx = int(w_idx_str)
            w, freq = idx2word_freq[w_idx]
            org_idx_l2_compact_idx.append(current_num_word_spacy - current_num_word_spacy_old)
            if w_idx in stop_ind_set:
                num_stop_words += 1
                continue
            if freq < min_freq:
                num_rare_words += 1
                continue
            if w_idx == EOS_IDX:
                continue
            #w_ind_spacy_corpus.append(w_idx)
            w_ind_spacy_corpus[current_num_word_spacy] = w_idx
            current_num_word_spacy += 1
            #valid_w_idx.append(w_idx)
        gpt2_to_spacy_inner = []
        for mapping_idx_str in gpt2_to_spacy.split():
            mapping_idx = int(mapping_idx_str)
            gpt2_to_spacy_inner.append( org_idx_l2_compact_idx[mapping_idx] + current_num_word_spacy_old )
        idx_gpt2_to_spacy[current_num_word_gpt2:current_num_word_gpt2+len(gpt2_idx)] = gpt2_to_spacy_inner
        current_num_word_gpt2 += len(gpt2_idx)
        #if current_sent == last_sent:
        #    num_duplicated_sent += 1
        #    continue
        #last_sent = current_sent
        #if len(fields) > max_sent_len:
        #    num_too_long_sent += 1
        #    continue
        if line_idx % 1000000 == 0:
            print(line_idx, )
            sys.stdout.flush()
        if line_idx >= max_sent_num:
            break
    
    w_ind_spacy_corpus_new, idx2word_freq_new = densify_index(w_ind_spacy_corpus, idx2word_freq)
    #print( "Finish loading {} sentences. While removing {} duplicated and {} long sentences".format(len(w_ind_corpus),num_duplicated_sent, num_too_long_sent) )
    #print(len(w_ind_gpt2_corpus))
    #print(len(idx_gpt2_to_spacy))
    #assert len(w_ind_gpt2_corpus) == len(idx_gpt2_to_spacy)
    print( "Remove {} stop words and {} rare words".format(num_stop_words, num_rare_words) )
    #print( "Finish loading {} sentences. Average gpt2 tokens {}. Average Spacy tokens {}.".format(line_idx, len(w_ind_gpt2_corpus)/float(line_idx), len(w_ind_spacy_corpus)/float(line_idx) ) )
    return w_ind_gpt2_corpus, w_ind_spacy_corpus_new, idx_gpt2_to_spacy, idx2word_freq_new


corpus_input_name = args.data + "corpus_index"
dictionary_input_name = args.data + "dictionary_index"

#with open(dictionary_input_name) as f_in:
#    w_d2_ind_freq, max_ind = utils.load_word_dict(f_in)

with open(dictionary_input_name) as f_in:
    idx2word_freq = utils.load_idx2word_freq(f_in)

max_ind = len(idx2word_freq)



with open(args.stop_word_file) as f_in:
    #stop_ind_set = convert_stop_to_ind(f_in, w_d2_ind_freq)
    stop_ind_set = convert_stop_to_ind_lower(f_in, idx2word_freq)

total_num_words = 0
total_num_words_spacy = 0
with open(corpus_input_name) as f_in:
    for line_idx, line in enumerate(f_in):
        sent_spacy, sent_gpt2, gpt2_to_spacy = line.rstrip().split('\t')
        total_num_words += len(sent_gpt2.split())
        for w_idx_str in sent_spacy.split():
            w_idx = int(w_idx_str)
            if w_idx in stop_ind_set:
                continue
            w, freq = idx2word_freq[w_idx]
            if freq < args.min_freq:
                continue
            if w_idx == EOS_IDX:
                continue
            total_num_words_spacy += 1
        if line_idx % 1000000 == 0:
            print(line_idx, )
            sys.stdout.flush()
        if line_idx >= args.max_sent_num:
            break

if total_num_words_spacy >= 2147483648:
    print("Will cause overflow")
    sys.exit()

with open(corpus_input_name) as f_in:
    w_ind_gpt2_corpus, w_ind_spacy_corpus_new, idx_gpt2_to_spacy, idx2word_freq_new = load_w_ind(f_in, args.max_sent_num, idx2word_freq, args.min_freq, stop_ind_set, total_num_words, total_num_words_spacy)

print("w_ind_gpt2_corpus[:50]", w_ind_gpt2_corpus[:50])
print("w_ind_spacy_corpus_new[:50]", w_ind_spacy_corpus_new[:50])
print("idx_gpt2_to_spacy[:50]", idx_gpt2_to_spacy[:50])
print("w_ind_gpt2_corpus[-50:]", w_ind_gpt2_corpus[-50:])
print("w_ind_spacy_corpus_new[-50:]", w_ind_spacy_corpus_new[-50:])
print("idx_gpt2_to_spacy[-50:]", idx_gpt2_to_spacy[-50:])

with open(args.new_dict_file, 'w') as f_out:
    for idx, (w, freq) in enumerate(idx2word_freq_new):
        f_out.write(w+'\t'+str(freq)+'\t'+str(idx) + '\n')
#args.max_target_num+args.max_sent_len
#print("Allocating {} bytes".format( (len(w_ind_gpt2_corpus)+len(w_ind_spacy_corpus_new)+len(idx_gpt2_to_spacy))*4 ) )

#print("Finish loading all files")
if idx_gpt2_to_spacy[-1] >= 2147483648:
    print("Will cause overflow")
    sys.exit()

training_output_name = args.save + "train.pt"
val_org_output_name = args.save + "val_org.pt"
test_org_output_name = args.save + "test_org.pt"

corpus_size = len(w_ind_gpt2_corpus)
testing_size = int(corpus_size * args.testing_ratio)
print("Testing size: {}".format(testing_size))

store_type = torch.int32

def store_tensors(f_out,w_ind_gpt2_corpus, w_ind_spacy_corpus_new, idx_gpt2_to_spacy, start_idx, end_idx):
    w_ind_gpt2_tensor = torch.tensor(w_ind_gpt2_corpus[start_idx:end_idx], dtype = store_type)
    idx_gpt2_to_spacy_tensor = torch.tensor(idx_gpt2_to_spacy[start_idx:end_idx], dtype = store_type)
    #print(idx_gpt2_to_spacy[start_idx:end_idx].dtype)
    #print(idx_gpt2_to_spacy_tensor.dtype)
    start_idx_spacy = int(idx_gpt2_to_spacy[start_idx])
    #print((idx_gpt2_to_spacy_tensor-start_idx_spacy).dtype)
    end_idx_spacy = int(idx_gpt2_to_spacy[end_idx-1])
    #print(start_idx_spacy, end_idx_spacy)
    w_ind_spacy_tensor = torch.tensor(w_ind_spacy_corpus_new[start_idx_spacy:end_idx_spacy], dtype = store_type)
    torch.save([w_ind_gpt2_tensor, w_ind_spacy_tensor, idx_gpt2_to_spacy_tensor-start_idx_spacy],f_out)

with open(test_org_output_name,'wb') as f_out:
    store_tensors(f_out, w_ind_gpt2_corpus, w_ind_spacy_corpus_new, idx_gpt2_to_spacy, -testing_size, corpus_size)

with open(val_org_output_name,'wb') as f_out:
    store_tensors(f_out, w_ind_gpt2_corpus, w_ind_spacy_corpus_new, idx_gpt2_to_spacy, -2*testing_size, -testing_size)

with open(training_output_name,'wb') as f_out:
    store_tensors(f_out, w_ind_gpt2_corpus, w_ind_spacy_corpus_new, idx_gpt2_to_spacy, 0, -2*testing_size)
