import sys
sys.path.insert(0, sys.path[0]+'/..')

import os
import argparse
import torch
import numpy as np

import utils_testing
import utils
from utils import str2bool
import colorama
import re

from gpt2_model.tokenization_gpt2 import GPT2Tokenizer
from gpt2_model.modeling_gpt2_condition import GPT2LMHeadModel
from gpt2_model.configuration_gpt2 import GPT2Config

import colorama



def create_args_parser():
	parser = argparse.ArgumentParser(description='PyTorch Interactive LM')
	#path
	parser.add_argument('--checkpoint_topics', type=str, default='../../models/',
						help='topical model checkpoint to use')
	parser.add_argument('--checkpoint_conditional', type=str, default='../../models/',
						help='conditional LM model checkpoint to use')
	parser.add_argument('--emb_file', type=str, default='target_emb.pt',
						help='path to a word embedding file')
	parser.add_argument('--word_dict', type=str, default='../../data/processed/wiki2016_gpt2/tensors_all_min100/dict_idx_compact',
						help='path to a dictionary file')
	#parser.add_argument('--outf', type=str, default='../gen_log/generated.txt',
	#                    help='output file for generated text')

	#parser.add_argument('--batch_size', type=int, default=3, metavar='N',
	#                    help='batch size')
	parser.add_argument('--num_sent_gen', type=int, default=3, metavar='N',
						help='In each prompt, generate how many sentences')
	parser.add_argument('--gen_sent_len', type=int, default=50, metavar='N',
						help='In each prompt, generate sentences with length gen_sent_len')
	parser.add_argument('--bptt', type=int, default=512,
						help='sequence length')
	parser.add_argument('--bptt_conditional', type=int, default=256,
						help='sequence length')
	parser.add_argument('--top_k_nn', type=int, default=5,
						help='Representing each topic using how many words')

	parser.add_argument('--cuda_topics', type=str2bool, nargs='?', default=True,
						help='use CUDA for topical model')
	parser.add_argument('--cuda_conditional', type=str2bool, nargs='?', default=True,
						help='use CUDA for conditional LM')
	parser.add_argument('--single_gpu', default=True, action='store_true',
						help='use single GPU')

	utils_testing.add_model_arguments(parser)
	return parser


def load_all_components(args):
	if args.emb_file == "target_emb.pt":
		args.emb_file =  os.path.join(args.checkpoint_topics,"target_emb.pt")
	device_topics = torch.device("cuda" if args.cuda_topics else "cpu")
	device_conditional = torch.device("cuda" if args.cuda_conditional else "cpu")
	#device_topics = torch.device("cuda:0" if args.cuda_topics else "cpu")
	#device_conditional = torch.device("cuda:1" if args.cuda_conditional else "cpu")
	with open(args.word_dict) as f_in:
		idx2word_freq = utils.load_idx2word_freq(f_in)
	word_d2_idx = {}
	for i in range(len(idx2word_freq)):
		w, freq = idx2word_freq[i]
		word_d2_idx[w] = i

	parallel_encoder, parallel_decoder, encoder, decoder, word_norm_emb = utils.loading_all_models(args, idx2word_freq, device_topics)
	output_emb_size = word_norm_emb.size(1)
	#print(next(encoder.parameters()).device)

	model_name = 'gpt2'

	encoder_state_dict = torch.load(os.path.join(args.checkpoint_conditional, 'encoder.pt'), map_location=device_conditional)
	gpt2_config = GPT2Config.from_pretrained(model_name)
	gpt2_config.word_emb_dim = output_emb_size
	model_condition = GPT2LMHeadModel.from_pretrained(model_name, state_dict = encoder_state_dict, config = gpt2_config).cuda(device_conditional)
	#print(next(model_condition.parameters()).device)

	tokenizer_GPT2 = GPT2Tokenizer.from_pretrained('distilgpt2')

	encoder.eval()
	decoder.eval()
	model_condition.eval()
	return idx2word_freq, word_d2_idx, parallel_encoder, parallel_decoder, word_norm_emb, model_condition, tokenizer_GPT2, device_topics, device_conditional

def show_future_topics(prompt, encoder, decoder, word_norm_emb, n_basis, top_k, bptt, idx2word_freq, tokenizer_GPT2, device_topics):
    tokenized_text = tokenizer_GPT2.tokenize(prompt, add_prefix_space=True)
    #print(tokenized_text)
    indexed_tokens = tokenizer_GPT2.convert_tokens_to_ids(tokenized_text)
    start_idx = len(indexed_tokens) - bptt
    if start_idx > 0:
        indexed_tokens = indexed_tokens[start_idx:]
    feature = torch.tensor(indexed_tokens, dtype=torch.long, device=device_topics).unsqueeze(0)
    output_emb, past = encoder(feature)
    output_emb_last = output_emb[:,-1,:]
    basis_pred = decoder(output_emb_last)
    basis_norm_pred = basis_pred / (0.000000000001 + basis_pred.norm(dim = 2, keepdim=True) )

    basis_norm_pred = basis_norm_pred.permute(0,2,1)
    sim_pairwise = torch.matmul(word_norm_emb.unsqueeze(dim = 0), basis_norm_pred)
    top_value, top_index = torch.topk(sim_pairwise, top_k, dim = 1, sorted=True)
    top_value = top_value / (0.000000000001 + top_value.sum(dim = 1, keepdim=True) )
    #out_str = ''
    for j in range(n_basis):
        out_str = str(j) + ', '
        for k in range(top_k):
        #for k in range(3):
            word_nn = idx2word_freq[top_index[0,k,j].item()][0]
            #out_str += word_nn+' {:5.3f} '.format(top_value[0,k,j].item())
            out_str += word_nn+', '
        print(out_str)
    print()

    return top_value, top_index, feature

def conditional_generation(selected_conditions, gen_sent_len, num_sent_gen, word_d2_idx, idx2word_freq, model_condition, word_norm_emb, top_index, top_value, feature, bptt_conditional, tokenizer_GPT2, device_conditional):
    word_norm_emb_top = word_norm_emb[top_index,:]
    word_norm_emb_w_sum = torch.sum( word_norm_emb_top * top_value.unsqueeze(-1), dim = 1) / top_value.unsqueeze(-1).sum(dim = 1)
    word_w_sum_norm = word_norm_emb_w_sum / (0.000000000001 + word_norm_emb_w_sum.norm(dim = -1, keepdim=True))
    word_w_sum_norm = word_w_sum_norm.to(device=device_conditional)
    selected_topic_idx = []
    selected_word_idx = []
    for x in selected_conditions:
        if isinstance(x, int):
            selected_topic_idx.append(x)
        else:
            if x not in word_d2_idx:
                print('Warning: Ignore the word '+x+' because it is too rare')
                continue
            selected_word_idx.append(word_d2_idx[x])
    selected_topic_idx = torch.tensor(np.sort(selected_topic_idx), dtype=torch.long, device = device_conditional)
    selected_word_idx = torch.tensor(selected_word_idx, dtype=torch.long, device = device_conditional)

    end_int = feature.size(1)
    max_prompt_len = bptt_conditional - gen_sent_len
    start_int = 0
    if end_int > max_prompt_len:
        start_int = end_int - max_prompt_len
    insert_loc_list = []
    insert_loc_list.append(end_int - 1)
    insert_loc_truncated = np.array(insert_loc_list) - start_int

    feature_expanded = feature[0,start_int:end_int].unsqueeze(0).expand(num_sent_gen,end_int - start_int).to(device = device_conditional)
    future_emb_chosen_topics = word_w_sum_norm[0, selected_topic_idx,:]
    future_emb_chosen_words = word_norm_emb[selected_word_idx,:]
    num_selection = future_emb_chosen_topics.size(0) + future_emb_chosen_words.size(0)
    future_emb_chosen = torch.cat([future_emb_chosen_topics, future_emb_chosen_words],dim=0).unsqueeze(0).expand(num_sent_gen,num_selection,word_norm_emb.size(-1))
    future_emb_chosen_arr = []
    future_emb_chosen_arr.append(future_emb_chosen)
    truncate_idx = 0
    output = utils_testing.sample_seq(model_condition, feature_expanded, insert_loc_truncated[truncate_idx:], future_emb_chosen_arr[truncate_idx:], gen_sent_len, device_conditional)
    output_org = utils_testing.sample_seq(model_condition, feature_expanded, None, None, gen_sent_len, device_conditional)
    
    print(colorama.Fore.BLUE+"Prompt: "+tokenizer_GPT2.decode(feature[0,start_int:end_int])+'\n'+colorama.Style.RESET_ALL)
    for j in range(num_sent_gen):
        generated_sent = tokenizer_GPT2.convert_tokens_to_string( [tokenizer_GPT2._convert_id_to_token(x) for x in output[j, :].tolist()] )
        generated_sent = generated_sent.replace('â',"'").replace('â','-').replace('\n'," ")
        utils_testing.print_sampled_sent(selected_topic_idx.tolist(), generated_sent, top_index[0,:,:], idx2word_freq, sys.stdout, 'conditional '+ str(j), selected_word_idx.tolist())
    
    print("\n"+colorama.Fore.BLUE+"Prompt: "+tokenizer_GPT2.decode(feature[0,start_int:end_int])+'\n'+colorama.Style.RESET_ALL)
    for j in range(num_sent_gen):
        generated_sent_org = tokenizer_GPT2.convert_tokens_to_string( [tokenizer_GPT2._convert_id_to_token(x) for x in output_org[j, :].tolist()] )
        generated_sent_org = generated_sent_org.replace('â',"'").replace('â','-').replace('\n'," ")
        utils_testing.print_sampled_sent(selected_topic_idx.tolist(), generated_sent_org, top_index[0,:,:], idx2word_freq, sys.stdout, 'original '+ str(j), selected_word_idx.tolist())
