
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


def print_basis_text(feature, idx2word_freq, top_value, top_index, i_batch, outf, idx_l2_w_gpt2, inner_idx_tensor):
    #n_basis = coeff_order.shape[1]
    batch_size_num_head, top_k, n_basis = top_index.size()
    batch_size, num_head = inner_idx_tensor.size()
    top_index = top_index.view(batch_size, num_head, top_k, n_basis)
    top_value = top_value.view(batch_size, num_head, top_k, n_basis)
    feature_text = convert_feature_to_text(feature, idx_l2_w_gpt2)
    #for i_sent in range(len(feature_text)):
    for i_sent in range(batch_size):
        #outf.write('{} batch, {}th sent: '.format(i_batch, i_sent)+' '.join(feature_text[i_sent])+'\n')
        last_end = -1
        for m in range(num_head):
            end = inner_idx_tensor[i_sent,m].item()
            if end == last_end:
                continue
            last_end = end
            outf.write(''.join(feature_text[i_sent][:end]).replace('Ġ',' ')+'\n')

            for j in range(n_basis):
                #org_ind = coeff_order[i_sent, j]
                #outf.write(str(j)+', org '+str(org_ind)+', '+str( coeff_sum[i_sent,org_ind,0] )+' - '+str( coeff_sum[i_sent,org_ind,1] )+': ')

                for k in range(top_k):
                    word_nn = idx2word_freq[top_index[i_sent,m,k,j].item()][0]
                    outf.write( word_nn+' {:5.3f}'.format(top_value[i_sent,m,k,j].item())+' ' )
                outf.write('\n')
            outf.write('\n')

def visualize_topics_val(dataloader, parallel_encoder, parallel_decoder, word_norm_emb, idx2word_freq, outf, n_basis, max_batch_num, de_en_connection, idx_l2_w_gpt2):
    #topics_num = 0
    top_k = 5
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dataloader):
            feature, target_unfold, inner_idx_tensor, future_mask = sample_batched

            basis_norm_pred, top_value, top_index = predict_batch(feature, inner_idx_tensor, future_mask, parallel_encoder, parallel_decoder, word_norm_emb, n_basis, top_k, de_en_connection)
            print_basis_text(feature, idx2word_freq, top_value, top_index, i_batch, outf, idx_l2_w_gpt2, inner_idx_tensor)

            if i_batch >= max_batch_num:
                break
