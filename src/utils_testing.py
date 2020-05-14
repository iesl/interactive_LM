import spacy
import torch
from spacy.lang.en import English
#import nsd_loss
import numpy as np
#from scipy.spatial import distance
import gc
import sys
import torch.utils.data
import json
import math
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from utils import str2bool
#sys.path.insert(0, sys.path[0]+'/testing/sim')
import math
import random
import re
import colorama
import run_pplm
from ngram import ngram
from result_statistics import result_statistics
from topic_result_statistics import topic_result_statistics
from sklearn.cluster import KMeans
from gensim.models import ldamodel
from model import SparseCoding
import time

def add_model_arguments(parser):
    ###decoder
    #parser.add_argument('--de_model', type=str, default='LSTM',
    parser.add_argument('--de_model', type=str, default='TRANS',
                        help='type of decoder model (LSTM, LSTM+TRANS, TRANS+LSTM, TRANS)')
    parser.add_argument('--trans_layers', type=int, default=5,
                        help='How many layers we have in transformer. Do not have effect if de_model is LSTM')
    parser.add_argument('--de_en_connection', type=str2bool, nargs='?', default=False,
                        help='If True, using Transformer decoder in our decoder. Otherwise, using Transformer encoder')
    parser.add_argument('--nhidlast2', type=int, default=300,
                        help='hidden embedding size of the second LSTM')
    parser.add_argument('--n_basis', type=int, default=10,
                        help='number of basis we want to predict')
    parser.add_argument('--positional_option', type=str, default='linear',
                        help='options of encode positional embedding into models (linear, cat, add)')
    parser.add_argument('--dropoutp', type=float, default=0,
                        help='dropout of positional embedding or input embedding after linear transformation (when linear_mapping_dim != 0)')
    parser.add_argument('--dropout_prob_trans', type=float, default=0,
                    help='hidden_dropout_prob and attention_probs_dropout_prob in Transformer')

def compute_freq_prob_idx2word(idx2word_freq):
    all_word, all_freq= list( zip(*idx2word_freq) )
    freq_sum = float(sum(all_freq))
    for i, (w, freq) in enumerate(idx2word_freq):
        idx2word_freq[i].append(freq/freq_sum)

def compute_freq_prob(word_d2_idx_freq):
    all_idx, all_freq= list( zip(*word_d2_idx_freq.values()) )
    freq_sum = float(sum(all_freq))
    for w in word_d2_idx_freq:
        idx, freq = word_d2_idx_freq[w]
        word_d2_idx_freq[w].append(freq/freq_sum)

def load_gpt2_vocab(f_in):
    w_d2_idx = json.load(f_in)
    max_w_idx = max(w_d2_idx.values()) 
    idx_l2_w_gpt2 = ['' for i in range(max_w_idx+1)]
    for w in w_d2_idx:
        idx = w_d2_idx[w]
        idx_l2_w_gpt2[idx] = w
    return idx_l2_w_gpt2

def predict_batch_simple(feature, inner_idx_tensor, future_mask, parallel_encoder, parallel_decoder, de_en_connection):
    output_emb, past = parallel_encoder(feature)
    hidden_size = output_emb.size(2)
    batch_size, num_head, seq_len = future_mask.size()
    output_emb_head = output_emb.gather(dim=1, index=inner_idx_tensor.unsqueeze(dim=-1).expand(batch_size,num_head,hidden_size))
    output_emb_last = output_emb_head.view(-1, output_emb_head.size(2))
    if de_en_connection:
        output_emb_masked = output_emb.unsqueeze(dim=1).expand(batch_size,num_head,seq_len,hidden_size)
        output_emb_masked = output_emb_masked.reshape(-1, seq_len, hidden_size)
        future_mask = future_mask.view(-1, seq_len)

        basis_pred = parallel_decoder(output_emb_last, output_emb_masked, memory_attention_mask = future_mask)
    else:
        basis_pred = parallel_decoder(output_emb_last)

    basis_norm_pred = basis_pred / (0.000000000001 + basis_pred.norm(dim = 2, keepdim=True) )
    return basis_norm_pred

def predict_batch(feature, inner_idx_tensor, future_mask, parallel_encoder, parallel_decoder, word_norm_emb, n_basis, top_k, de_en_connection):
    basis_norm_pred = predict_batch_simple(feature, inner_idx_tensor, future_mask, parallel_encoder, parallel_decoder, de_en_connection)
    basis_norm_pred = basis_norm_pred.permute(0,2,1)
    sim_pairwise = torch.matmul(word_norm_emb.unsqueeze(dim = 0), basis_norm_pred)
    top_value, top_index = torch.topk(sim_pairwise, top_k, dim = 1, sorted=True)
    return basis_norm_pred, top_value, top_index

def convert_feature_to_text(feature, idx_l2_w_gpt2):
    feature_list = feature.tolist()
    feature_text = []
    for i in range(feature.size(0)):
        current_sent = []
        for w_ind in feature_list[i]:
            w = idx_l2_w_gpt2[w_ind]
            current_sent.append(w)
        feature_text.append(current_sent)
    return feature_text


#def print_basis_text(feature, idx2word_freq, top_value, top_index, i_batch, outf, idx_l2_w_gpt2, inner_idx_tensor):
def print_basis_text(feature, idx2word_freq, top_value_arr, top_index_arr, method_name_arr, i_batch, outf, tokenizer_GPT2, inner_idx_tensor):
    #n_basis = coeff_order.shape[1]
    batch_size, num_head, top_k, n_basis = top_index_arr[0].size()
    batch_size, num_head = inner_idx_tensor.size()
    #feature_text = convert_feature_to_text(feature, idx_l2_w_gpt2)
    feature_text = [ [tokenizer_GPT2._convert_id_to_token(x) for x in feature[i,:].tolist()] for i in range(feature.size(0))]
    #print(feature_text)
    #for i_sent in range(len(feature_text)):
    for i_sent in range(batch_size):
        #outf.write('{} batch, {}th sent: '.format(i_batch, i_sent)+' '.join(feature_text[i_sent])+'\n')
        last_end = -1
        for m in range(num_head):
            end = inner_idx_tensor[i_sent,m].item()
            if end == last_end:
                continue
            last_end = end
            #outf.write(''.join(feature_text[i_sent][:end]).replace('Ġ',' ')+'\n')
            outf.write(tokenizer_GPT2.convert_tokens_to_string(feature_text[i_sent][:end])+'\n\n')

            for q, method_name in enumerate(method_name_arr):
                top_index = top_index_arr[q] #.view(batch_size, num_head, top_k, n_basis)
                top_value = top_value_arr[q] #.view(batch_size, num_head, top_k, n_basis)
                outf.write( method_name + ': \n' )
                for j in range(n_basis):
                    #org_ind = coeff_order[i_sent, j]
                    #outf.write(str(j)+', org '+str(org_ind)+', '+str( coeff_sum[i_sent,org_ind,0] )+' - '+str( coeff_sum[i_sent,org_ind,1] )+': ')

                    for k in range(top_k):
                        word_nn = idx2word_freq[top_index[i_sent,m,k,j].item()][0]
                        outf.write( word_nn+' {:5.3f}'.format(top_value[i_sent,m,k,j].item())+' ' )
                    outf.write('\n')
                outf.write('\n')

#def visualize_topics_val(dataloader, parallel_encoder, parallel_decoder, word_norm_emb, idx2word_freq, outf, n_basis, max_batch_num, de_en_connection, idx_l2_w_gpt2):
def visualize_topics_val(dataloader, parallel_encoder, parallel_decoder, word_norm_emb, idx2word_freq, outf, n_basis, max_batch_num, de_en_connection, tokenizer_GPT2):
    #topics_num = 0
    top_k = 5
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dataloader):
            if i_batch >= max_batch_num:
                break
            feature, target_unfold, inner_idx_tensor, future_mask = sample_batched

            basis_norm_pred, top_value, top_index = predict_batch(feature, inner_idx_tensor, future_mask, parallel_encoder, parallel_decoder, word_norm_emb, n_basis, top_k, de_en_connection)
            #print_basis_text(feature, idx2word_freq, top_value, top_index, i_batch, outf, idx_l2_w_gpt2, inner_idx_tensor)
            batch_size, num_head = inner_idx_tensor.size()
            print_basis_text(feature, idx2word_freq, [top_value.view(batch_size, num_head, top_k, n_basis)], [top_index.view(batch_size, num_head, top_k, n_basis)], ['NSD'], i_batch, outf, tokenizer_GPT2, inner_idx_tensor)



def print_sampled_sent(selected_topic_idx, generated_sent, top_index_im, idx2word_freq, outf, print_prefix, selected_word_idx=None):
    def insert_substring(sent, insert_loc, insert_substring):
        return sent[:insert_loc] + insert_substring + sent[insert_loc:]
    
    def highlight_words(word_nn, generated_sent, topic_l2_word_d2_count_t):
        def find_all(a_str, sub):
            start = 0
            while True:
                start = a_str.find(sub, start)
                if start == -1: return
                yield start
                start += len(sub) # use start += 1 to find overlapping matches
        index_shift = 0

        #for m in re.finditer(word_nn, generated_sent):
        #    start = m.start() + index_shift
        #    end = m.end() + index_shift
        for m_start in find_all(generated_sent, word_nn):
            start = m_start + index_shift
            end = m_start + len(word_nn) + index_shift
            #print(generated_sent)
            #print(word_nn)
            #print(start, end)
            #if start != 0 and generated_sent[start-1] != ' ' and end >= len(generated_sent) - 1 and generated_sent[end+1] != ' ':
            if end < len(generated_sent) - 1 and generated_sent[end+1] != ' ' and start != 0 and generated_sent[start-1] != ' ': 
                continue
            if word_nn not in topic_l2_word_d2_count_t:
                topic_l2_word_d2_count_t[word_nn] = 0
            topic_l2_word_d2_count_t[word_nn] += 1
            prev_start = generated_sent[:start].rfind(colorama.Fore.RED)
            prev_end = generated_sent[:start].rfind(colorama.Style.RESET_ALL)
            if prev_start > prev_end:
                continue
            generated_sent = insert_substring(generated_sent, end, colorama.Style.RESET_ALL)
            generated_sent = insert_substring(generated_sent, start, colorama.Fore.RED)
            index_shift += len(colorama.Style.RESET_ALL) + len(colorama.Fore.RED)
        return generated_sent
        
    num_selected = len(selected_topic_idx)
    top_k = top_index_im.size(0)
    topic_l2_word_d2_count = [{} for t in range(num_selected)]
    for t in range(num_selected):
        topic_idx = selected_topic_idx[t]
        for k in range(top_k):
            #word_nn = idx2word_freq[top_index[i_sent,m,k,topic_idx].item()][0]
            #print(top_index_im.size())
            #print(topic_idx)

            word_nn = idx2word_freq[top_index_im[k,topic_idx].item()][0]
            generated_sent = highlight_words(word_nn, generated_sent, topic_l2_word_d2_count[t])
    num_word = 0
    if selected_word_idx is not None:
        num_word = len(selected_word_idx)
        word_l2_word_d2_count = [{} for t in range(num_word)]
        for t in range(num_word):
            word_nn = idx2word_freq[selected_word_idx[t]][0]
            generated_sent = highlight_words(word_nn, generated_sent, word_l2_word_d2_count[t])
    outf.write(print_prefix + ': ' + generated_sent + '\n')
    for t in range(num_selected):
        if len(topic_l2_word_d2_count[t]) == 0:
            continue
        topic_idx = selected_topic_idx[t]
        outf.write(str(topic_idx)+' topic: '+str(topic_l2_word_d2_count[t])+'\n')
    for t in range(num_word):
        if len(word_l2_word_d2_count[t]) == 0:
            continue
        outf.write('word: '+str(word_l2_word_d2_count[t])+'\n')
    outf.write('\n')


def print_basis_conditional_text(feature, pplm_sent, idx2word_freq, top_value, top_index, i_batch, outf, tokenizer_GPT2, inner_idx_tensor, gen_sent_tensor, gen_sent_tensor_org, selected_topic_idx_arr, gpt2_model, result_stats):
    batch_size, num_head, top_k, n_basis = top_index.size()
    num_sent_gen = gen_sent_tensor.size(2)
    #feature_text = [ [tokenizer_GPT2._convert_id_to_token(x) for x in feature[i,:].tolist()] for i in range(feature.size(0))]
    # for i_sent in range(1):
    for i_sent in range(batch_size):
        outf.write('batch number: ' + str(i_sent) + '\n')
        last_end = -1
        # for m in range(1):
        for m in range(num_head):
            outf.write('number of head: ' + str(m) + '\n')
            end = inner_idx_tensor[i_sent,m].item()
            if end == last_end:
                continue
            last_end = end
            #outf.write(tokenizer_GPT2.convert_tokens_to_string(feature_text[i_sent][:end])+'\n')
            outf.write(tokenizer_GPT2.decode(feature[i_sent,:end])+'\n')
            
            for j in range(n_basis):

                #org_ind = coeff_order[i_sent, j]
                #outf.write(str(j)+', org '+str(org_ind)+', '+str( coeff_sum[i_sent,org_ind,0] )+' - '+str( coeff_sum[i_sent,org_ind,1] )+': ')
                outf.write( str(j) + ', ' )
                for k in range(top_k):
                    #print(i_sent,m,k,j, top_index.size())
                    #print(top_index[i_sent,m,k,j].item(), len(idx2word_freq))
                    word_nn = idx2word_freq[top_index[i_sent,m,k,j].item()][0]
                    outf.write( word_nn+' {:5.3f}'.format(top_value[i_sent,m,k,j].item())+', ' )
                outf.write('\n')
            outf.write('\n')
            selected_topic_idx = selected_topic_idx_arr[i_sent][m]
            outf.write('Select these topics '+' '.join([str(x) for x in selected_topic_idx])+'\n')

            if len(pplm_sent[i_sent][m][0]) == 0:
                outf.write('Skipping this context because PPLM cannot condition on any word.\n')
                continue

            for j in range(num_sent_gen):
                #During the print, highlight the words which occur in generated sentences
                #search directly without tokenization
                #make this a function
                #generated_sent = tokenizer_GPT2.convert_tokens_to_string( [tokenizer_GPT2._convert_id_to_token(x) for x in gen_sent_tensor[i_sent, m, j, :].tolist()] )
                generated_sent = tokenizer_GPT2.decode( gen_sent_tensor[i_sent, m, j, :] )
                print_sampled_sent(selected_topic_idx, generated_sent, top_index[i_sent,m,:,:], idx2word_freq, outf, 'conditional '+ str(j))
                result_stats.update("Model condition", gen_sent_tensor[i_sent, m, j, :], feature[i_sent,:end], selected_topic_idx, top_index[i_sent,m,:,:], idx2word_freq, tokenizer_GPT2)
            if gen_sent_tensor_org.size(0) > 0:
                for j in range(num_sent_gen):
                    #generated_sent_org = tokenizer_GPT2.convert_tokens_to_string( [tokenizer_GPT2._convert_id_to_token(x) for x in gen_sent_tensor_org[i_sent, m, j, :].tolist()] )
                    generated_sent_org = tokenizer_GPT2.decode( gen_sent_tensor_org[i_sent, m, j, :] )
                    print_sampled_sent(selected_topic_idx, generated_sent_org, top_index[i_sent,m,:,:], idx2word_freq, outf, 'original '+ str(j))
                    result_stats.update("Original", gen_sent_tensor_org[i_sent, m, j, :], feature[i_sent,:end], selected_topic_idx, top_index[i_sent,m,:,:], idx2word_freq, tokenizer_GPT2)
            for j in range(num_sent_gen):
                sentence = torch.tensor(tokenizer_GPT2.encode(pplm_sent[i_sent][m][j]), device="cuda", dtype=torch.long)
                print_sampled_sent(selected_topic_idx, pplm_sent[i_sent][m][j], top_index[i_sent,m,:,:], idx2word_freq, outf, 'pplm model '+ str(j))
                result_stats.update("PPLM", sentence, feature[i_sent,:end], selected_topic_idx, top_index[i_sent,m,:,:], idx2word_freq, tokenizer_GPT2)
            outf.write('\n\n')
        result_stats.renew_ngram()

def top_k_logits(logits, k):
    #modified from https://github.com/graykode/gpt-2-Pytorch/blob/master/GPT2/sample.py
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1].view(-1, 1).expand_as(logits)
    return torch.where(logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits)

def sample_seq(model_condition, context, insert_loc, future_emb_chosen_arr, gen_sent_len, device, temperature=1, top_k = 40, sample=True):
    #modified from https://github.com/graykode/gpt-2-Pytorch/blob/master/GPT2/sample.py
    prev = context
    batch_size = prev.size(0)
    output = torch.zeros((batch_size, 0), dtype=torch.long, device = device  )
    past = None
    #sample = False
    for i in range(gen_sent_len):
        if i == 0:
            outputs_condition = model_condition(prev, past=past, insert_loc=insert_loc, future_emb_chosen_arr=future_emb_chosen_arr)  # lm_logits, presents, (all hidden_states), (attentions)
        else:
            outputs_condition = model_condition(prev, past=past)
        logits = outputs_condition[0]
        past = outputs_condition[1]
        logits = logits[:, -1, :] / temperature
        logits = top_k_logits(logits, k=top_k)
        #max_val, _ = torch.max(logits,dim=1,keepdim=True)
        #logits_max_too_small = (max_val < -50)
        #logits[logits_max_too_small * (logits != -1e10 )] = 0
        #exp_logits = torch.exp(logits)
        #probs = exp_logits / torch.sum(exp_logits, dim = -1, keepdim=True)
        
        #if de.nonzero().size(0) != de.numel():
        #    print(logits)
        #    sys.exit(0)
        probs = F.softmax(logits, dim=-1)
        #probs = torch.exp(F.log_softmax(logits, dim=-1))
        if sample:
            #probs[probs < 0] = 0
            if torch.isnan(probs).sum() > 0:
                print(past)
                print(prev)
                print(insert_loc)
                print(future_emb_chosen_arr)
                print(logits)
                print(probs)
                sys.exit(0)
            prev = torch.multinomial(probs, num_samples=1)
        else:
            _, prev = torch.topk(probs, k=1, dim=-1)
        output = torch.cat((output, prev), dim=1)
    return output


def visualize_interactive_LM(model_condition, pplm_model, gpt2_model, device_conditional, num_sent_gen, gen_sent_len, dataloader, parallel_encoder, parallel_decoder, word_norm_emb, idx2word_freq, outf, n_basis, max_batch_num, de_en_connection, tokenizer_GPT2, bptt_conditional):
    top_k = 5
    with torch.no_grad():
        result_stats = result_statistics(gpt2_model)
        result_stats.add_model("Model condition")
        result_stats.add_model("Original")
        result_stats.add_model("PPLM")
        for i_batch, sample_batched in enumerate(dataloader):
            # if i_batch == 0:
            #     continue
            print("batch"+str(i_batch))
            sys.stdout.flush()
            feature, target_unfold, inner_idx_tensor, future_mask = sample_batched
            feature_text = [ [tokenizer_GPT2._convert_id_to_token(x) for x in feature[i,:].tolist()] for i in range(feature.size(0))]

            basis_norm_pred, top_value, top_index = predict_batch(feature, inner_idx_tensor, future_mask, parallel_encoder, parallel_decoder, word_norm_emb, n_basis, top_k, de_en_connection)
            batch_size, num_head = inner_idx_tensor.size()

            # the index of each words in the vocab list
            top_index = top_index.view(batch_size, num_head, top_k, n_basis)
            # the value of each words
            top_value = top_value.view(batch_size, num_head, top_k, n_basis)

            word_norm_emb_top = word_norm_emb[top_index,:]
            word_norm_emb_w_sum = torch.sum( word_norm_emb_top * top_value.unsqueeze(-1), dim = 2) / top_value.unsqueeze(-1).sum(dim = 2)
            word_w_sum_norm = word_norm_emb_w_sum / (0.000000000001 + word_norm_emb_w_sum.norm(dim = -1, keepdim=True))
            word_w_sum_norm = word_w_sum_norm.to(device=device_conditional)
            gen_sent_tensor = torch.empty( (batch_size, num_head, num_sent_gen, gen_sent_len), dtype=torch.long, device=device_conditional )
            gen_sent_tensor_org = torch.empty( (batch_size, num_head, num_sent_gen, gen_sent_len), dtype=torch.long, device=device_conditional )
            selected_topic_idx_arr =[ [[] for j in range(num_head)] for i in range(batch_size)]
            pplm_sent = []

            # for i_sent in range(1):
            for i_sent in range(batch_size):
                print("sent"+str(i_sent))
                insert_loc_list = []
                future_emb_chosen_arr = []
                last_end = -1
                temp = []
                # for m in range(1):
                for m in range(num_head):
                    print("head"+str(m))

                    end = inner_idx_tensor[i_sent,m]
                    if end == last_end:
                        continue
                    last_end = end
                    end_int = end.item()
                    max_prompt_len = bptt_conditional - gen_sent_len
                    start_int = 0
                    if end_int > max_prompt_len:
                        start_int = end_int - max_prompt_len
                    insert_loc_list.append(end_int - 1)
                    num_selection = random.randint(1, n_basis)
                    selected_topic_idx = np.sort(np.random.choice(n_basis, size=num_selection, replace = False))
                    selected_topic_idx_arr[i_sent][m] = selected_topic_idx.tolist()
                    # generate bag-of-words
                    bag_of_words = []
                    for each in selected_topic_idx.tolist():
                        for k in range(top_k):
                            word_nn = idx2word_freq[top_index[i_sent,m,k,each].item()][0]
                            bag_of_words.append(word_nn)
                    
                    t = time.time()
                    selected_topic_idx = torch.tensor(selected_topic_idx, dtype=torch.long, device = device_conditional)
                    feature_expanded = feature[i_sent,start_int:end].unsqueeze(0).expand(num_sent_gen,end_int - start_int).to(device = device_conditional)
                    future_emb_chosen = word_w_sum_norm[i_sent, m, selected_topic_idx,:].unsqueeze(0).expand(num_sent_gen,num_selection,word_norm_emb.size(-1))
                    future_emb_chosen_arr.append(future_emb_chosen)
                    insert_loc_truncated = np.array(insert_loc_list) - start_int
                    #truncate_idx = 0
                    #while( insert_loc_truncated[truncate_idx] < 0 ):
                    #    truncate_idx += 1
                    truncate_idx = -1
                    output = sample_seq(model_condition, feature_expanded, insert_loc_truncated[truncate_idx:], future_emb_chosen_arr[truncate_idx:], gen_sent_len, device_conditional,)
                    condition_elapsed = time.time() - t
                    gen_sent_tensor[i_sent, m, :, :] = output
                    
                    t = time.time()
                    output_org = sample_seq(model_condition, feature_expanded, insert_loc_truncated[truncate_idx:], [torch.zeros( (num_sent_gen,0,word_norm_emb.size(-1)),device = device_conditional) for x in range(insert_loc_truncated[truncate_idx:].size)], gen_sent_len, device_conditional,)
                    #output_org = sample_seq(model_condition, feature_expanded, None, None, gen_sent_len, device_conditional,)
                    org_elapsed = time.time() - t
                    gen_sent_tensor_org[i_sent, m, :, :] = output_org
                    
                    #gen_text = tokenizer_GPT2.decode(output)
                    t = time.time()
                    context = tokenizer_GPT2.convert_tokens_to_string(feature_text[i_sent][:end])
                    try:
                        gen_text, _ = pplm_model.run_pplm_example(context, False, num_sent_gen, bag_of_words, gen_sent_len, 0.05, 1.0, top_k, True, 1, 10000, 1, 0, False, 1.5, 0.9, 0.01, True)
                    except:
                        print("Skipping {} batch, {} paragraph, and {} head because PPLM cannot condition on any word".format(i_batch,i_sent,m))
                        gen_text = ['']*num_sent_gen
                        #print(context)
                        #print(num_sent_gen)
                        #print(bag_of_words)
                        #print(gen_sent_len)
                        #sys.exit(1)
                        #print(gen_text,perplexity)
                        #gen_text, _ = pplm_model.run_pplm_example(context, False, num_sent_gen, bag_of_words, gen_sent_len, 0.05, 1.0, top_k, True, 1, 10000, 1, 0, False, 1.5, 0.9, 0.01, True)
                    pplm_elapsed = time.time() - t
                    temp.append(gen_text)
                    
                    if len(gen_text[0]) > 0:
                        #result_stats.model_results["time_count"] += 1
                        for method_name, time_spent in [ ("Model condition",condition_elapsed), ("Original", org_elapsed), ("PPLM",pplm_elapsed) ]:
                            result_stats.model_results[method_name]["time_sum"] += time_spent
                            result_stats.model_results[method_name]["time_count"] += 1

                pplm_sent.append(temp)
            print_basis_conditional_text(feature, pplm_sent, idx2word_freq, top_value, top_index, i_batch, outf, tokenizer_GPT2, inner_idx_tensor, gen_sent_tensor, gen_sent_tensor_org, selected_topic_idx_arr, gpt2_model, result_stats)
            #result_stats.renew_ngram()
            if i_batch + 1 >= max_batch_num:
                break
        result_stats.print()
        result_stats.generate_report(outf)

def get_word_list_spacy(inner_idx_tensor, feature_text, tokenizer_GPT2, nlp, word_d2_idx, stop_word_set, OOV_set):
    def get_word_lest_from_text(feature_text_i):
        feature_text_i_str = tokenizer_GPT2.convert_tokens_to_string(feature_text_i)
        tokens = nlp.tokenizer(feature_text_i_str)
        word_idx_list_i_j = []
        for tok in tokens:
            w = tok.text
            #print(w, )
            if w not in word_d2_idx:
                continue
            w_idx = word_d2_idx[w]
            if w_idx in stop_word_set or w_idx in OOV_set:
                continue
            word_idx_list_i_j.append(w_idx)
        return word_idx_list_i_j
    word_idx_list = []
    word_idx_rest_list = []
    batch_size, num_head = inner_idx_tensor.size()
    inner_idx_tensor_np = inner_idx_tensor.cpu().numpy()
    for b, feature_text_i in enumerate(feature_text):
        word_idx_list_i = []
        word_idx_rest_list_i = []
        for j in range(num_head):
            end_idx = inner_idx_tensor_np[b,j]
            word_idx_list_i_j = get_word_lest_from_text(feature_text_i[:end_idx])
            #assert len(word_idx_list_i_j) > 0, print(feature_text_i[:end_idx])
            word_idx_list_i.append(word_idx_list_i_j)
            if end_idx == len(feature_text_i):
                word_idx_rest_list_i.append([])
            else:
                word_idx_rest_list_i_j = get_word_lest_from_text(feature_text_i[end_idx:])
                word_idx_rest_list_i.append(word_idx_rest_list_i_j)
            #count = word_idx_d2_count.get(w_idx,0)
            #word_idx_d2_count[w_idx] += 1
        word_idx_list.append(word_idx_list_i)
        word_idx_rest_list.append(word_idx_rest_list_i)
    return word_idx_list, word_idx_rest_list

def get_topic_emb(basis_norm_pred, word_norm_emb, top_k, batch_size, num_head):
    n_basis = basis_norm_pred.size(1)
    sim_pairwise = torch.matmul(word_norm_emb.unsqueeze(dim = 0), basis_norm_pred.permute(0,2,1))
    top_value, top_index = torch.topk(sim_pairwise, top_k, dim = 1, sorted=True)
    # the index of each words in the vocab list
    top_index = top_index.view(batch_size, num_head, top_k, n_basis)
    # the value of each words
    top_value = top_value.view(batch_size, num_head, top_k, n_basis)
    
    word_norm_emb_top = word_norm_emb[top_index,:]
    word_norm_emb_w_sum = torch.sum( word_norm_emb_top * top_value.unsqueeze(-1), dim = 2)/ top_value.unsqueeze(-1).sum(dim = 2)
    word_w_sum_norm = word_norm_emb_w_sum / (0.000000000001 + word_norm_emb_w_sum.norm(dim = -1, keepdim=True))
    return top_value, top_index, word_w_sum_norm

def random_vocab_sampling(could_sample_list, word_norm_emb, batch_size, num_head, n_basis, top_k):
    idx_chosen = np.empty([batch_size, num_head, n_basis], dtype=int)
    for b in range(batch_size):
        for j in range(num_head):
            idx_chosen[b,j,:] = np.random.choice(could_sample_list, n_basis, replace=False)
    basis_norm_pred = word_norm_emb[idx_chosen.reshape(batch_size * num_head, n_basis),:]
    top_value, top_index, word_w_sum_norm = get_topic_emb(basis_norm_pred, word_norm_emb, top_k, batch_size, num_head)
    return top_value, top_index, word_w_sum_norm

def random_word_sampling(word_idx_list, word_norm_emb, n_basis, top_k):
    batch_size = len(word_idx_list)
    num_head = len(word_idx_list[0])
    idx_chosen = np.zeros([batch_size, num_head, n_basis], dtype=int)
    for b in range(batch_size):
        for j in range(num_head):
            #end_idx = inner_idx_tensor_np[b,j]
            if len(word_idx_list[b][j]) <= 1:
                continue
            if len(word_idx_list[b][j]) < n_basis:
                idx_chosen[b,j,:] = np.random.choice(word_idx_list[b][j], n_basis, replace=True)
            else:
                idx_chosen[b,j,:] = np.random.choice(word_idx_list[b][j], n_basis, replace=False)

    #idx_chosen = []
    #for word_idx_list_i in word_idx_list:
    #    idx_chosen_i = np.random.choice(word_idx_list_i, n_basis, replace=True)
    #    idx_chosen.append(idx_chosen_i)
    basis_norm_pred = word_norm_emb[idx_chosen.reshape(batch_size * num_head, n_basis),:]
    top_value, top_index, word_w_sum_norm = get_topic_emb(basis_norm_pred, word_norm_emb, top_k, batch_size, num_head)
    return top_value, top_index, word_w_sum_norm

def SC_clustering(cluster_feature, n_basis, max_iter):
    with torch.enable_grad():
        lr = 0.1
        L1_losss_B = 0.2
        SC = SparseCoding(n_basis, cluster_feature.size(0), cluster_feature.size(1), device=cluster_feature.device)
        loss_func = torch.nn.MSELoss(reduction='sum')
        opt = torch.optim.RMSprop(SC.parameters(), lr=lr, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
        #print("\nloss: ")
        for i in range(max_iter):
            opt.zero_grad()
            pred = SC()
            loss = loss_func(pred, cluster_feature) / 2
            #print("{:5.2f}".format(loss.item()), end =" ") 
            # loss += L1_losss_B * mr.coeff.abs().sum()
            #loss += L1_losss_B * (mr.coeff.abs().sum() + mr.coeff.diagonal(dim1=1, dim2=2).abs().sum())
            loss += L1_losss_B * SC.coeff.abs().sum()
            # print('loss:', loss.item())
            loss.backward()
            opt.step()
            SC.compute_coeff_pos()

        return SC.code_book

def cluster_sampling(word_idx_list, word_norm_emb, n_basis, top_k, cluster_method="Kmeans"):
    batch_size = len(word_idx_list)
    num_head = len(word_idx_list[0])
    device_topic = word_norm_emb.device
    if cluster_method=="KMeans":
        kmeans_model = KMeans(n_clusters=n_basis, n_init=1, random_state=0, init='random')
    emb_size = word_norm_emb.size(-1)
    basis_pred = torch.zeros([batch_size, num_head, n_basis, emb_size], device=device_topic)
    for b in range(batch_size):
        for j in range(num_head):
            word_index_context = word_idx_list[b][j]
            if len(word_index_context) <= 1:
                continue
            if len(word_index_context) < n_basis:
                word_index_context += np.random.choice(word_index_context, n_basis - len(word_index_context), replace=True).tolist()
                basis_pred[b,j,:,:] = word_norm_emb[word_index_context,:]
            else:
                cluster_feature = word_norm_emb[word_index_context].cpu().numpy()
                if cluster_method=="KMeans":
                    kmeans_model.fit(cluster_feature)
                    basis_pred[b,j,:,:] = torch.tensor(kmeans_model.cluster_centers_,device=device_topic)
                elif cluster_method== "Sparse_coding":
                    #print(cluster_feature)
                    basis_pred[b,j,:,:] = SC_clustering(torch.tensor(cluster_feature,device=device_topic), n_basis, max_iter = 200)
    basis_pred = basis_pred.view(batch_size * num_head, n_basis, emb_size)
    basis_norm_pred = basis_pred / (0.000000000001 + basis_pred.norm(dim = 2, keepdim=True) )
    top_value, top_index, word_w_sum_norm = get_topic_emb(basis_norm_pred, word_norm_emb, top_k, batch_size, num_head)
    return top_value, top_index, word_w_sum_norm

def load_lda_model(LDA_model_path, word_d2_idx, word_norm_emb, top_k):
    lda_model = ldamodel.LdaModel.load(LDA_model_path, mmap='r')
    lda_word_prob_raw = lda_model.get_topics()
    lda_id2word = lda_model.id2word
    #id_lda_d2_id_dict = {}
    word_num, emb_size = word_norm_emb.size()
    lda_topic_num, num_vocab_lda = lda_word_prob_raw.shape
    lda_word_prob = np.zeros( (lda_topic_num, word_num) )
    for lda_id in lda_id2word:
        word = lda_id2word[lda_id]
        if word not in word_d2_idx:
            continue
        id_dict = word_d2_idx[word]
        lda_word_prob[:,id_dict] = lda_word_prob_raw[:,lda_id]
        #id_lda_d2_id_dict[lda_id] = id_dict
    lda_word_prob_tensor = torch.tensor(lda_word_prob, device = word_norm_emb.device, dtype=torch.float)
    
    lda_top_word_value, lda_top_word_index = torch.topk(lda_word_prob_tensor, top_k, dim = 1, sorted=True)
    top_word_emb = word_norm_emb[lda_top_word_index,:]
    #top_word_emb should have size (lda_topic_num, top_k, emb_size)
    top_word_emb_sum = torch.sum(top_word_emb * lda_top_word_value.unsqueeze(-1), dim = 1)
    lda_fixed_topic_emb = top_word_emb_sum/ (0.000000000001 + top_word_emb_sum.norm(dim = -1, keepdim=True) )
    #lda_top_word_value, lda_top_word_index, lda_fixed_topic_emb = get_topic_emb(lda_fixed_topic_emb, word_norm_emb, top_k, batch_size, num_head)
    
    sim_pairwise = torch.matmul(word_norm_emb, lda_fixed_topic_emb.permute(1,0))
    top_value, top_index = torch.topk(sim_pairwise, top_k, dim = 0, sorted=True)
    # the index of each words in the vocab list
    lda_top_word_index = top_index.permute(1,0)
    # the value of each words
    lda_top_word_value = top_value.permute(1,0)
    
    word_norm_emb_top = word_norm_emb[lda_top_word_index,:]
    word_norm_emb_w_sum = torch.sum( word_norm_emb_top * lda_top_word_value.unsqueeze(-1), dim = 1)/ lda_top_word_value.unsqueeze(-1).sum(dim = 1)
    lda_fixed_topic_emb = word_norm_emb_w_sum / (0.000000000001 + word_norm_emb_w_sum.norm(dim = -1, keepdim=True))
        
    return lda_word_prob_tensor, lda_top_word_value, lda_top_word_index, lda_fixed_topic_emb

def compose_context_emb(word_idx_list, word_norm_emb):
    batch_size = len(word_idx_list)
    num_head = len(word_idx_list[0])
    emb_size = word_norm_emb.size(1)
    context_emb = torch.zeros((batch_size, num_head, emb_size), device=word_norm_emb.device)
    for b in range(batch_size):
        for j in range(num_head):
            word_index_context = word_idx_list[b][j]
            if len(word_index_context) <= 1:
                continue
            context_emb[b,j,:] = word_norm_emb[word_index_context,:].sum(dim = 0)
    context_norm_emb = context_emb / (0.000000000001 + context_emb.norm(dim = -1, keepdim=True) )

    return context_norm_emb

def select_fixed_lda_topics(context_norm_emb, lda_top_word_value, lda_top_word_index, lda_fixed_topic_emb, n_basis):
    batch_size, num_head, emb_size = context_norm_emb.size()
    sim_pairwise = torch.mm(context_norm_emb.view(batch_size*num_head, emb_size), lda_fixed_topic_emb.permute(1,0))
    _, selected_topic_idx = torch.topk(sim_pairwise, n_basis, dim = 1, sorted=True)
    selected_topic_idx = selected_topic_idx.view(batch_size, num_head, n_basis)
    top_value = lda_top_word_value[selected_topic_idx,:].permute(0,1,3,2)
    top_index = lda_top_word_index[selected_topic_idx,:].permute(0,1,3,2)
    word_w_sum_norm = lda_fixed_topic_emb[selected_topic_idx,:]
    return top_value, top_index, word_w_sum_norm

def select_dynamic_lda_topics(context_norm_emb, word_norm_emb, lda_word_prob_tensor, n_basis, top_k):
    num_words = word_norm_emb.size(0)
    batch_size, num_head, emb_size = context_norm_emb.size()
    sim_pairwise = torch.mm(context_norm_emb.view(batch_size*num_head, emb_size), word_norm_emb.permute(1,0))
    #sim_pairwise should have size (batch_size*num_head, num_words)
    sim_pairwise = sim_pairwise.view(batch_size, num_head, num_words)
    #lda_word_prob_tensor should have size (lda_topic_num, num_words)
    #top_value = torch.zeros((batch_size, num_head, top_k, n_basis), device = word_norm_emb.device)
    #top_index = torch.zeros((batch_size, num_head, top_k, n_basis), dtype=torch.long, device = word_norm_emb.device)
    word_w_sum_norm = torch.zeros((batch_size, num_head, n_basis, emb_size), device = word_norm_emb.device)
    for b in range(batch_size):
        for j in range(num_head):
            lda_word_prob_local = lda_word_prob_tensor * sim_pairwise[b,j,:].unsqueeze(0)
            lda_top_word_value, lda_top_word_index = torch.topk(lda_word_prob_local, top_k, dim = 1, sorted=True)
            top_word_emb = word_norm_emb[lda_top_word_index,:]
            top_word_emb_sum = torch.sum(top_word_emb * lda_top_word_value.unsqueeze(-1), dim = 1)
            lda_dynamic_topic_emb = top_word_emb_sum/ (0.000000000001 + top_word_emb_sum.norm(dim = -1, keepdim=True) )

            sim_pairwise_bj = torch.mm(context_norm_emb[b,j,:].unsqueeze(dim=0), lda_dynamic_topic_emb.permute(1,0))
            _, selected_topic_idx = torch.topk(sim_pairwise_bj.squeeze(dim=0), n_basis, dim = 0, sorted=True)
            #print(lda_top_word_value)
            #print(selected_topic_idx)
            #top_value[b,j,:,:] = lda_top_word_value[selected_topic_idx,:].permute(1,0)
            #top_index[b,j,:,:] = lda_top_word_index[selected_topic_idx,:].permute(1,0)
            word_w_sum_norm[b,j,:,:] = lda_dynamic_topic_emb[selected_topic_idx,:]
    gc.collect()
    top_value, top_index, word_w_sum_norm = get_topic_emb(word_w_sum_norm.view(batch_size*num_head, n_basis, emb_size), word_norm_emb, top_k, batch_size, num_head)

    return top_value, top_index, word_w_sum_norm 

def testing_all_topic_baselines(dataloader, parallel_encoder, parallel_decoder, word_norm_emb, idx2word_freq, outf, n_basis, max_batch_num, tokenizer_GPT2, stop_word_set, topic_models, de_en_connection, LDA_model_path, word_emb_center_path):
    top_k = 5
    emb_sum = torch.sum(word_norm_emb,dim=1)
    OOV_list = torch.nonzero(emb_sum == 0).squeeze().cpu().tolist()
    print("OOV number = {}".format(len(OOV_list)))
    print("OOV index examples {}".format(OOV_list[:10]))
    OOV_set = set(OOV_list)
    could_sample_list = list(  set(list(range(len(idx2word_freq)))) - (OOV_set | stop_word_set) )
    if topic_models != 'random_vocab':
        nlp = English()
        word_d2_idx = {}
        for idx in range(len(idx2word_freq)):
            word = idx2word_freq[idx][0]
            word_d2_idx[word] = idx
    if 'LDA' in topic_models:
        lda_word_prob_tensor, lda_top_word_value, lda_top_word_index, lda_fixed_topic_emb = load_lda_model(LDA_model_path, word_d2_idx, word_norm_emb, top_k)
    if 'global_centers' in topic_models:
        word_emb_centers = np.loadtxt(word_emb_center_path)
        word_emb_centers = torch.tensor(word_emb_centers, dtype=torch.float, device = word_norm_emb.device)
        word_norm_emb_centers = word_emb_centers / (0.000000000001 + word_emb_centers.norm(dim = -1, keepdim=True) )
        sim_pairwise = torch.mm(word_norm_emb_centers, word_norm_emb.permute(1,0))
        w_emb_top_word_value, w_emb_top_word_index = torch.topk(sim_pairwise, top_k, dim = 1, sorted=True)

    with torch.no_grad():
        topic_result_stats = topic_result_statistics()
        for t_model in topic_models.split('+'):
            topic_result_stats.add_model(t_model)
        #topic_result_stats.add_model("Model condition")
        #result_stats.add_model("Original")
        #topic_result_stats.add_model("PPLM")
        for i_batch, sample_batched in enumerate(dataloader):
            # if i_batch == 0:
            #     continue
            if i_batch >= max_batch_num:
                break
            print("batch"+str(i_batch))
            sys.stdout.flush()
            feature, target_unfold, inner_idx_tensor, future_mask = sample_batched
            feature_text = [ [tokenizer_GPT2._convert_id_to_token(x) for x in feature[i,:].tolist()] for i in range(feature.size(0))]
            
            #tokenized_feature = [nlp(feature_text_i).text for feature_text_i in feature_text]
            #word_idx_d2_count = {}
            batch_size, num_head = inner_idx_tensor.size()

            top_value_arr = []
            top_index_arr = []
            method_name_arr = []
            if topic_models != 'random_vocab':
                word_idx_list, word_idx_rest_list = get_word_list_spacy(inner_idx_tensor, feature_text, tokenizer_GPT2, nlp, word_d2_idx, stop_word_set, OOV_set)
            
            if 'random_vocab' in  topic_models:
                top_value, top_index, word_w_sum_norm = random_vocab_sampling(could_sample_list, word_norm_emb, batch_size, num_head, n_basis, top_k)
                topic_result_stats.evaluate_topic_models("random_vocab", top_value, top_index, word_w_sum_norm, word_idx_list, word_idx_rest_list, idx2word_freq, word_norm_emb)
                method_name_arr.append('random_vocab') ; top_value_arr.append(top_value) ; top_index_arr.append(top_index)

            if 'random_word' in topic_models:
                top_value, top_index, word_w_sum_norm = random_word_sampling(word_idx_list, word_norm_emb, n_basis, top_k)
                topic_result_stats.evaluate_topic_models("random_word", top_value, top_index, word_w_sum_norm, word_idx_list, word_idx_rest_list, idx2word_freq, word_norm_emb)
                method_name_arr.append('random_word') ; top_value_arr.append(top_value) ; top_index_arr.append(top_index)

            if 'SC_cluster' in topic_models:
                top_value, top_index, word_w_sum_norm = cluster_sampling(word_idx_list, word_norm_emb, n_basis, top_k, cluster_method="Sparse_coding")
                topic_result_stats.evaluate_topic_models("SC_cluster", top_value, top_index, word_w_sum_norm, word_idx_list, word_idx_rest_list, idx2word_freq, word_norm_emb)
                method_name_arr.append('SC_cluster') ; top_value_arr.append(top_value) ; top_index_arr.append(top_index)
            
            if 'kmeans_cluster' in topic_models:
                top_value, top_index, word_w_sum_norm = cluster_sampling(word_idx_list, word_norm_emb, n_basis, top_k, cluster_method="KMeans")
                topic_result_stats.evaluate_topic_models("kmeans_cluster", top_value, top_index, word_w_sum_norm, word_idx_list, word_idx_rest_list, idx2word_freq, word_norm_emb)
                method_name_arr.append('kmeans_cluster') ; top_value_arr.append(top_value) ; top_index_arr.append(top_index)
            


            if 'LDA' in topic_models or 'global_centers' in topic_models:
                context_norm_emb = compose_context_emb(word_idx_list, word_norm_emb)

            if 'LDA_org' in topic_models:
                top_value, top_index, word_w_sum_norm = select_fixed_lda_topics(context_norm_emb, lda_top_word_value, lda_top_word_index, lda_fixed_topic_emb, n_basis)
                topic_result_stats.evaluate_topic_models("LDA_org", top_value, top_index, word_w_sum_norm, word_idx_list, word_idx_rest_list, idx2word_freq, word_norm_emb)
                method_name_arr.append('LDA_org') ; top_value_arr.append(top_value) ; top_index_arr.append(top_index)
            
            if 'LDA_plus' in topic_models:
                top_value, top_index, word_w_sum_norm = select_dynamic_lda_topics(context_norm_emb, word_norm_emb, lda_word_prob_tensor, n_basis, top_k)
                topic_result_stats.evaluate_topic_models("LDA_plus", top_value, top_index, word_w_sum_norm, word_idx_list, word_idx_rest_list, idx2word_freq, word_norm_emb)
                method_name_arr.append('LDA_plus') ; top_value_arr.append(top_value) ; top_index_arr.append(top_index)
            
            if 'global_centers' in topic_models:
                top_value, top_index, word_w_sum_norm = select_fixed_lda_topics(context_norm_emb, w_emb_top_word_value, w_emb_top_word_index, word_norm_emb_centers, n_basis)
                topic_result_stats.evaluate_topic_models("global_centers", top_value, top_index, word_w_sum_norm, word_idx_list, word_idx_rest_list, idx2word_freq, word_norm_emb)
                method_name_arr.append('global_centers') ; top_value_arr.append(top_value) ; top_index_arr.append(top_index)
            
            if 'NSD' in topic_models:
                basis_norm_pred, top_value, top_index = predict_batch(feature, inner_idx_tensor, future_mask, parallel_encoder, parallel_decoder, word_norm_emb, n_basis, top_k, de_en_connection)
                # the index of each words in the vocab list
                top_index = top_index.view(batch_size, num_head, top_k, n_basis)
                # the value of each words
                top_value = top_value.view(batch_size, num_head, top_k, n_basis)
                
                word_norm_emb_top = word_norm_emb[top_index,:]
                word_norm_emb_w_sum = torch.sum( word_norm_emb_top * top_value.unsqueeze(-1), dim = 2)/ top_value.unsqueeze(-1).sum(dim = 2)
                word_w_sum_norm = word_norm_emb_w_sum / (0.000000000001 + word_norm_emb_w_sum.norm(dim = -1, keepdim=True))
                topic_result_stats.evaluate_topic_models("NSD", top_value, top_index, word_w_sum_norm, word_idx_list, word_idx_rest_list, idx2word_freq, word_norm_emb)
                method_name_arr.append('NSD') ; top_value_arr.append(top_value) ; top_index_arr.append(top_index)

            print_basis_text(feature, idx2word_freq, top_value_arr, top_index_arr, method_name_arr, i_batch, outf, tokenizer_GPT2, inner_idx_tensor)

        topic_result_stats.generate_report(outf)
        topic_result_stats.generate_report(sys.stdout)


def testing_topic_baseline(model_condition, pplm_model, gpt2_model, device_conditional, num_sent_gen, gen_sent_len, dataloader, word_norm_emb, idx2word_freq, outf, n_basis, max_batch_num, tokenizer_GPT2, bptt_conditional, topic_mode, stop_word_set):
    top_k = 5
    nlp = English()
    word_d2_idx = {}
    for idx in range(len(idx2word_freq)):
        word = idx2word_freq[idx][0]
        word_d2_idx[word] = idx
        
    emb_sum = torch.sum(word_norm_emb,dim=1)
    OOV_list = torch.nonzero(emb_sum == 0).squeeze().cpu().tolist()
    print("OOV number = {}".format(len(OOV_list)))
    print("OOV index examples {}".format(OOV_list[:10]))
    OOV_set = set(OOV_list)
    if topic_mode == 'random_vocab':
        could_sample_list = list(  set(list(range(len(idx2word_freq)))) - (OOV_set | stop_word_set) )
    with torch.no_grad():
        result_stats = result_statistics(gpt2_model)
        result_stats.add_model("Model condition")
        #result_stats.add_model("Original")
        result_stats.add_model("PPLM")
        for i_batch, sample_batched in enumerate(dataloader):
            # if i_batch == 0:
            #     continue
            print("batch"+str(i_batch))
            sys.stdout.flush()
            feature, target_unfold, inner_idx_tensor, future_mask = sample_batched
            feature_text = [ [tokenizer_GPT2._convert_id_to_token(x) for x in feature[i,:].tolist()] for i in range(feature.size(0))]
            
            #tokenized_feature = [nlp(feature_text_i).text for feature_text_i in feature_text]
            #word_idx_d2_count = {}
            batch_size, num_head = inner_idx_tensor.size()

            if topic_mode != 'random_vocab':
                word_idx_list, word_idx_rest_list = get_word_list_spacy(inner_idx_tensor, feature_text, tokenizer_GPT2, nlp, word_d2_idx, stop_word_set, OOV_set)
            
            if topic_mode == 'random_vocab':
                top_value, top_index, word_w_sum_norm = random_vocab_sampling(could_sample_list, word_norm_emb, batch_size, num_head, n_basis, top_k)

            elif topic_mode == 'random_word':
                top_value, top_index, word_w_sum_norm = random_word_sampling(word_idx_list, word_norm_emb, n_basis, top_k)

            elif topic_mode == 'cluster':
                top_value, top_index, word_w_sum_norm = cluster_sampling(word_idx_list, word_norm_emb, n_basis, top_k)

            elif topic_mode == 'LDA':
                pass
            #basis_norm_pred, top_value, top_index = predict_batch(feature, inner_idx_tensor, future_mask, parallel_encoder, parallel_decoder, word_norm_emb, n_basis, top_k, de_en_connection)
            word_w_sum_norm = word_w_sum_norm.to(device=device_conditional)

            gen_sent_tensor = torch.empty( (batch_size, num_head, num_sent_gen, gen_sent_len), dtype=torch.long, device=device_conditional )
            #gen_sent_tensor_org = torch.empty( (batch_size, num_head, num_sent_gen, gen_sent_len), dtype=torch.long, device=device_conditional )
            gen_sent_tensor_org = torch.zeros(0)
            selected_topic_idx_arr =[ [[] for j in range(num_head)] for i in range(batch_size)]
            pplm_sent = []

            # for i_sent in range(1):
            for i_sent in range(batch_size):
                print("sent"+str(i_sent))
                #if i_sent == 0:
                #    continue
                insert_loc_list = []
                future_emb_chosen_arr = []
                last_end = -1
                temp = []
                # for m in range(1):
                for m in range(num_head):
                    print("head"+str(m))
                    #if m <4:
                    #    continue
                    end = inner_idx_tensor[i_sent,m]
                    if end == last_end:
                        continue
                    last_end = end
                    end_int = end.item()
                    max_prompt_len = bptt_conditional - gen_sent_len
                    start_int = 0
                    if end_int > max_prompt_len:
                        start_int = end_int - max_prompt_len
                    insert_loc_list.append(end_int - 1)
                    num_selection = random.randint(1, n_basis)
                    selected_topic_idx = np.sort(np.random.choice(n_basis, size=num_selection, replace = False))
                    selected_topic_idx_arr[i_sent][m] = selected_topic_idx.tolist()
                    # generate bag-of-words
                    bag_of_words = []
                    for each in selected_topic_idx.tolist():
                        for k in range(top_k):
                            word_nn = idx2word_freq[top_index[i_sent,m,k,each].item()][0]
                            bag_of_words.append(word_nn)
                        pass
                    
                    context = tokenizer_GPT2.convert_tokens_to_string(feature_text[i_sent][:end])

                    selected_topic_idx = torch.tensor(selected_topic_idx, dtype=torch.long, device = device_conditional)
                    feature_expanded = feature[i_sent,start_int:end].unsqueeze(0).expand(num_sent_gen,end_int - start_int).to(device = device_conditional)
                    future_emb_chosen = word_w_sum_norm[i_sent, m, selected_topic_idx,:].unsqueeze(0).expand(num_sent_gen,num_selection,word_norm_emb.size(-1))
                    future_emb_chosen_arr.append(future_emb_chosen)
                    insert_loc_truncated = np.array(insert_loc_list) - start_int
                    #truncate_idx = 0
                    #while( insert_loc_truncated[truncate_idx] < 0 ):
                    #    truncate_idx += 1
                    truncate_idx = -1
                    output = sample_seq(model_condition, feature_expanded, insert_loc_truncated[truncate_idx:], future_emb_chosen_arr[truncate_idx:], gen_sent_len, device_conditional,)
                    gen_sent_tensor[i_sent, m, :, :] = output
                    
                    gen_text = tokenizer_GPT2.decode(output)
                    #try:
                    #    gen_text, _ = pplm_model.run_pplm_example(context, False, num_sent_gen, bag_of_words, gen_sent_len, 0.05, 1.0, top_k, True, 1, 10000, 1, 0, False, 1.5, 0.9, 0.01, True)
                    #except:
                    #    print(context)
                    #    print(num_sent_gen)
                    #    print(bag_of_words)
                    #    print(gen_sent_len)
                    #    sys.exit(1)
                    #    print(gen_text,perplexity)
                    #    gen_text, _ = pplm_model.run_pplm_example(context, False, num_sent_gen, bag_of_words, gen_sent_len, 0.05, 1.0, top_k, True, 1, 10000, 1, 0, False, 1.5, 0.9, 0.01, True)
                    temp.append(gen_text)
                    #output_org = sample_seq(model_condition, feature_expanded, None, None, gen_sent_len, device_conditional,)
                    #gen_sent_tensor_org[i_sent, m, :, :] = output_org
                pplm_sent.append(temp)
            print_basis_conditional_text(feature, pplm_sent, idx2word_freq, top_value, top_index, i_batch, outf, tokenizer_GPT2, inner_idx_tensor, gen_sent_tensor, gen_sent_tensor_org, selected_topic_idx_arr, gpt2_model, result_stats)
            result_stats.renew_ngram()
            if i_batch + 1 >= max_batch_num:
                break
        result_stats.print()
        result_stats.generate_report(outf)
        
