import torch
import nsd_loss
import numpy as np
from scipy.spatial import distance
import gc
import sys
import torch.utils.data
import json
import torch.nn.functional as F
from utils import str2bool
sys.path.insert(0, sys.path[0]+'/testing/sim')
import math
import random
import re
import colorama

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
def print_basis_text(feature, idx2word_freq, top_value, top_index, i_batch, outf, tokenizer_GPT2, inner_idx_tensor):
    #n_basis = coeff_order.shape[1]
    batch_size_num_head, top_k, n_basis = top_index.size()
    batch_size, num_head = inner_idx_tensor.size()
    top_index = top_index.view(batch_size, num_head, top_k, n_basis)
    top_value = top_value.view(batch_size, num_head, top_k, n_basis)
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
            outf.write(tokenizer_GPT2.convert_tokens_to_string(feature_text[i_sent][:end])+'\n')

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
            feature, target_unfold, inner_idx_tensor, future_mask = sample_batched

            basis_norm_pred, top_value, top_index = predict_batch(feature, inner_idx_tensor, future_mask, parallel_encoder, parallel_decoder, word_norm_emb, n_basis, top_k, de_en_connection)
            #print_basis_text(feature, idx2word_freq, top_value, top_index, i_batch, outf, idx_l2_w_gpt2, inner_idx_tensor)
            print_basis_text(feature, idx2word_freq, top_value, top_index, i_batch, outf, tokenizer_GPT2, inner_idx_tensor)

            if i_batch >= max_batch_num:
                break

def insert_substring(sent, insert_loc, insert_substring):
    return sent[:insert_loc] + insert_substring + sent[insert_loc:]

def print_sampled_sent(selected_topic_idx, generated_sent, top_index_im, idx2word_freq, outf, print_prefix):
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
            index_shift = 0
            for m in re.finditer(word_nn, generated_sent):
                start = m.start() + index_shift
                end = m.end() + index_shift
                #print(generated_sent)
                #print(word_nn)
                #print(start, end)
                #if start != 0 and generated_sent[start-1] != ' ' and end >= len(generated_sent) - 1 and generated_sent[end+1] != ' ':
                if end < len(generated_sent) - 1 and generated_sent[end+1] != ' ' and start != 0 and generated_sent[start-1] != ' ': 
                    continue
                if word_nn not in topic_l2_word_d2_count[t]:
                    topic_l2_word_d2_count[t][word_nn] = 0
                topic_l2_word_d2_count[t][word_nn] += 1
                prev_start = generated_sent[:start].rfind(colorama.Fore.RED)
                prev_end = generated_sent[:start].rfind(colorama.Style.RESET_ALL)
                if prev_start > prev_end:
                    continue
                generated_sent = insert_substring(generated_sent, end, colorama.Style.RESET_ALL)
                generated_sent = insert_substring(generated_sent, start, colorama.Fore.RED)
                index_shift += len(colorama.Style.RESET_ALL) + len(colorama.Fore.RED)
    outf.write(print_prefix + ': ' + generated_sent + '\n')
    for t in range(num_selected):
        if len(topic_l2_word_d2_count[t]) == 0:
            continue
        topic_idx = selected_topic_idx[t]
        outf.write(str(topic_idx)+' topic: '+str(topic_l2_word_d2_count[t])+'\n')
    outf.write('\n')

def print_basis_conditional_text(feature, idx2word_freq, top_value, top_index, i_batch, outf, tokenizer_GPT2, inner_idx_tensor, gen_sent_tensor, gen_sent_tensor_org, selected_topic_idx_arr):
    #n_basis = coeff_order.shape[1]
    batch_size, num_head, top_k, n_basis = top_index.size()
    #batch_size, num_head = inner_idx_tensor.size()
    num_sent_gen = gen_sent_tensor.size(2)
    feature_text = [ [tokenizer_GPT2._convert_id_to_token(x) for x in feature[i,:].tolist()] for i in range(feature.size(0))]
    for i_sent in range(batch_size):
        last_end = -1
        for m in range(num_head):
            end = inner_idx_tensor[i_sent,m].item()
            if end == last_end:
                continue
            last_end = end
            outf.write(tokenizer_GPT2.convert_tokens_to_string(feature_text[i_sent][:end])+'\n')
            
            for j in range(n_basis):
                #org_ind = coeff_order[i_sent, j]
                #outf.write(str(j)+', org '+str(org_ind)+', '+str( coeff_sum[i_sent,org_ind,0] )+' - '+str( coeff_sum[i_sent,org_ind,1] )+': ')
                outf.write( str(j) + ', ' )
                for k in range(top_k):
                    word_nn = idx2word_freq[top_index[i_sent,m,k,j].item()][0]
                    outf.write( word_nn+' {:5.3f}'.format(top_value[i_sent,m,k,j].item())+' ' )
                outf.write('\n')
            outf.write('\n')
            selected_topic_idx = selected_topic_idx_arr[i_sent][m]
            outf.write('Select these topics '+' '.join([str(x) for x in selected_topic_idx])+'\n')
            for j in range(num_sent_gen):
                #During the print, highlight the words which occur in generated sentences
                #search directly without tokenization
                #make this a function
                generated_sent = tokenizer_GPT2.convert_tokens_to_string( [tokenizer_GPT2._convert_id_to_token(x) for x in gen_sent_tensor[i_sent, m, j, :].tolist()] )
                print_sampled_sent(selected_topic_idx, generated_sent, top_index[i_sent,m,:,:], idx2word_freq, outf, 'conditional '+ str(j))
            for j in range(num_sent_gen):
                generated_sent_org = tokenizer_GPT2.convert_tokens_to_string( [tokenizer_GPT2._convert_id_to_token(x) for x in gen_sent_tensor_org[i_sent, m, j, :].tolist()] )
                print_sampled_sent(selected_topic_idx, generated_sent_org, top_index[i_sent,m,:,:], idx2word_freq, outf, 'original '+ str(j))
            outf.write('\n\n')
    

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
    for i in range(gen_sent_len):
        if i == 0:
            outputs_condition = model_condition(prev, past=past, insert_loc=insert_loc, future_emb_chosen_arr=future_emb_chosen_arr)  # lm_logits, presents, (all hidden_states), (attentions)
        else:
            outputs_condition = model_condition(prev, past=past)
        logits = outputs_condition[0]
        past = outputs_condition[1]
        logits = logits[:, -1, :] / temperature
        logits = top_k_logits(logits, k=top_k)
        log_probs = F.softmax(logits, dim=-1)
        if sample:
            prev = torch.multinomial(log_probs, num_samples=1)
        else:
            _, prev = torch.topk(log_probs, k=1, dim=-1)
        output = torch.cat((output, prev), dim=1)
    return output

def visualize_interactive_LM(model_condition, device_conditional, num_sent_gen, gen_sent_len, dataloader, parallel_encoder, parallel_decoder, word_norm_emb, idx2word_freq, outf, n_basis, max_batch_num, de_en_connection, tokenizer_GPT2):
    #topics_num = 0
    top_k = 5
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dataloader):
            print(i_batch, )
            sys.stdout.flush()
            feature, target_unfold, inner_idx_tensor, future_mask = sample_batched

            basis_norm_pred, top_value, top_index = predict_batch(feature, inner_idx_tensor, future_mask, parallel_encoder, parallel_decoder, word_norm_emb, n_basis, top_k, de_en_connection)
            batch_size, num_head = inner_idx_tensor.size()
            top_index = top_index.view(batch_size, num_head, top_k, n_basis)
            top_value = top_value.view(batch_size, num_head, top_k, n_basis)
            word_norm_emb_top = word_norm_emb[top_index,:]
            word_norm_emb_w_sum = torch.sum( word_norm_emb_top * top_value.unsqueeze(-1), dim = 2) / top_value.unsqueeze(-1).sum(dim = 2)
            word_w_sum_norm = word_norm_emb_w_sum / (0.000000000001 + word_norm_emb_w_sum.norm(dim = -1, keepdim=True))
            word_w_sum_norm = word_w_sum_norm.to(device=device_conditional)
            #word_norm_emb_w_sum should have size (batch_size, num_head, n_basis, word emb size)
            gen_sent_tensor = torch.empty( (batch_size, num_head, num_sent_gen, gen_sent_len), dtype=torch.long, device=device_conditional )
            gen_sent_tensor_org = torch.empty( (batch_size, num_head, num_sent_gen, gen_sent_len), dtype=torch.long, device=device_conditional )
            selected_topic_idx_arr =[ [[] for j in range(num_head)] for i in range(batch_size)]
            for i_sent in range(batch_size):
                insert_loc_list = []
                future_emb_chosen_arr = []
                last_end = -1
                for m in range(num_head):
                    end = inner_idx_tensor[i_sent,m]
                    if end == last_end:
                        continue
                    last_end = end
                    end_int = end.item()
                    insert_loc_list.append(end_int - 1)
                    num_selection = random.randint(1, n_basis)
                    selected_topic_idx = np.sort(np.random.choice(n_basis, size=num_selection, replace = False))
                    selected_topic_idx_arr[i_sent][m] = selected_topic_idx.tolist()
                    selected_topic_idx = torch.tensor(selected_topic_idx, dtype=torch.long, device = device_conditional)
                    #compose_conditional_emb(top_value, top_index, word_norm_emb)
                    feature_expanded = feature[i_sent,:end].unsqueeze(0).expand(num_sent_gen,end_int).to(device = device_conditional)
                    future_emb_chosen = word_w_sum_norm[i_sent, m, selected_topic_idx,:].unsqueeze(0).expand(num_sent_gen,num_selection,word_norm_emb.size(-1))
                    future_emb_chosen_arr.append(future_emb_chosen)
                    output = sample_seq(model_condition, feature_expanded, np.array(insert_loc_list), future_emb_chosen_arr, gen_sent_len, device_conditional)
                    #print(output.size())
                    #print(gen_sent_tensor.size())
                    gen_sent_tensor[i_sent, m, :, :] = output
                    output_org = sample_seq(model_condition, feature_expanded, None, None, gen_sent_len, device_conditional)
                    gen_sent_tensor_org[i_sent, m, :, :] = output_org
            print_basis_conditional_text(feature, idx2word_freq, top_value, top_index, i_batch, outf, tokenizer_GPT2, inner_idx_tensor, gen_sent_tensor, gen_sent_tensor_org, selected_topic_idx_arr)

            if i_batch >= max_batch_num:
                break
