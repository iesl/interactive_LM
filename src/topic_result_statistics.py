import torch
import os
import math
import numpy as np
import utils_testing

class topic_result_statistics:

    def __init__(self):
        self.model_results = {}

    def add_model(self, model_name):
        self.model_results[model_name] = {}
        self.model_results[model_name]["batch count"] = 0
        self.model_results[model_name]["count"] = 0
        self.model_results[model_name]["topics_diversity_dist"] = 0
        self.model_results[model_name]["top_words_diversity_dist"] = 0
        self.model_results[model_name]["specificity"] = 0
        self.model_results[model_name]["novelty"] = 0
        self.model_results[model_name]["novelty_word"] = 0
        self.model_results[model_name]["novelty_word_w"] = 0
        self.model_results[model_name]["relevancy_f_25"] = 0
        self.model_results[model_name]["relevancy_f_25_arr"] = []
        self.model_results[model_name]["context_len_arr"] = []
        self.model_results[model_name]["relevancy_f_25_w"] = 0
        self.model_results[model_name]["relevancy_topic_f_25"] = 0
        self.model_results[model_name]["relevancy_f_25_count"] = 0
        self.model_results[model_name]["relevancy_f_25_count_arr"] = []
        self.model_results[model_name]["relevancy_f_25_count_w"] = 0
        self.model_results[model_name]["relevancy_f_all"] = 0
        self.model_results[model_name]["relevancy_f_all_count"] = 0

    
    def evaluate_topic_models(self, model_name, top_value, top_index, word_w_sum_norm, word_idx_list, word_idx_rest_list, idx2word_freq, word_norm_emb):
        def _emb_var(topic_embedding):
            n_basis = topic_embedding.size(0)
            topic_embedding_mean = topic_embedding.mean(dim = 0, keepdim = True)
            topic_sq = (topic_embedding - topic_embedding_mean) * (topic_embedding - topic_embedding_mean)
            return torch.sqrt(topic_sq.sum(dim = 1)).mean().item()

        def compute_word_weight(word_index, idx2word_freq, device):
            alpha = 1e-4
            num_word = len(word_index)
            word_weight_context = torch.zeros(num_word, device = device)
            for w_i in range(num_word):
                w_idx = word_index[w_i]
                prob = idx2word_freq[w_idx][2]
                word_weight_context[w_i] = alpha / (alpha + prob)
            return word_weight_context
        
        batch_size, num_head, top_k, n_basis = top_value.size()
        emb_size = word_w_sum_norm.size(-1)
        for i in range(batch_size):
            for j in range(num_head):
                word_index_future = word_idx_rest_list[i][j]
                word_index_context = word_idx_list[i][j]
                if len( word_index_context ) <= 1:
                    continue
                if len( word_index_future ) <= 1:
                    #skip all the results at the end of each paragraph, because we cannot evaluate the relevency
                    #when evaluating the relevancy, we skip the first future token because it might be an incompleted word (a word piece)
                    continue
                self.model_results[model_name]['count'] += 1
                topic_embedding = word_w_sum_norm[i,j,:,:]
                assert torch.nonzero(torch.sum(topic_embedding,dim=1) == 0).size(0) == 0 
                self.model_results[model_name]['topics_diversity_dist'] += _emb_var(topic_embedding)
                top_index_ij_np = top_index[i,j,:,:].cpu().numpy()
                specificity = 0
                #t_word_emb = np.empty( (top_k, n_basis, emb_size) )
                t_word_emb = torch.empty( (top_k, n_basis, emb_size) )
                for k in range(top_k):
                    for n in range(n_basis):
                        specificity += 1.0 / idx2word_freq[top_index_ij_np[k,n]][1]
                        t_word_emb[k,n,:] = word_norm_emb[top_index_ij_np[k,n],:]
                #t_word_emb_tensor = torch.tensor(t_word_emb.reshape(top_k * n_basis, emb_size))
                self.model_results[model_name]['top_words_diversity_dist'] += _emb_var(t_word_emb.view(top_k * n_basis, emb_size))
                self.model_results[model_name]['specificity'] += specificity / (top_k * n_basis)

                word_emb_context = word_norm_emb[word_index_context[:-1],:]
                word_weight_context = compute_word_weight(word_index_context[:-1], idx2word_freq, word_norm_emb.device)

                topic_word_sim = torch.mm(topic_embedding, word_emb_context.permute(1,0))
                topic_word_sim_max_topic, _ = topic_word_sim.max(dim = 1)
                topic_word_sim_max_word, _ = topic_word_sim.max(dim = 0)
                self.model_results[model_name]['novelty'] += 1 - topic_word_sim_max_topic.mean().item()
                self.model_results[model_name]['novelty_word'] += 1 - topic_word_sim_max_word.mean().item()
                self.model_results[model_name]['novelty_word_w'] += 1 - ((topic_word_sim_max_word*word_weight_context).sum()/word_weight_context.sum()).item()

                word_emb_future = word_norm_emb[word_index_future[1:],:]
                word_weight_future = compute_word_weight(word_index_future[1:], idx2word_freq, word_norm_emb.device)
                
                topic_future_word_sim = torch.mm(topic_embedding, word_emb_future.permute(1,0))
                topic_future_word_sim_max_word, _ = topic_future_word_sim.max(dim = 0)
                #print(topic_future_word_sim_max.size())
                #print(min(25,topic_future_word_sim_max.size(0)))
                future_window_size = 25
                if topic_future_word_sim_max_word.size(0) > future_window_size:
                    self.model_results[model_name]['relevancy_f_25_count'] += future_window_size
                    self.model_results[model_name]['relevancy_f_25'] += topic_future_word_sim_max_word[:future_window_size].sum().item()
                    if j >= len(self.model_results[model_name]['relevancy_f_25_count_arr']):
                        self.model_results[model_name]['relevancy_f_25_count_arr'].append(0)
                        self.model_results[model_name]['relevancy_f_25_arr'].append(0)
                        self.model_results[model_name]['context_len_arr'].append(0)
                    self.model_results[model_name]['relevancy_f_25_arr'][j] += topic_future_word_sim_max_word[:future_window_size].mean().item()
                    self.model_results[model_name]['relevancy_f_25_count_arr'][j] += 1
                    self.model_results[model_name]['context_len_arr'][j] += len(word_index_context[:-1])

                    self.model_results[model_name]['relevancy_f_25_count_w'] += 1
                    self.model_results[model_name]['relevancy_f_25_w'] += ( (topic_future_word_sim_max_word[:future_window_size]*word_weight_future[:future_window_size] ).sum() / word_weight_future[:future_window_size].sum() ).item()

                    topic_future_word_sim_max_topic, _ = topic_future_word_sim[:,:future_window_size].max(dim = 1)
                    self.model_results[model_name]['relevancy_topic_f_25'] += topic_future_word_sim_max_topic.sum().item()
                
                self.model_results[model_name]['relevancy_f_all_count'] += topic_future_word_sim_max_word.size(0)
                self.model_results[model_name]['relevancy_f_all'] += topic_future_word_sim_max_word.sum().item()


    def generate_report(self, outf):
        outf.write('Reports: \n')
        for model_name, model in self.model_results.items():
            outf.write(model_name + " count: " + str( model["count"]) + '\n')
            outf.write(model_name + " topics_diversity_dist: " + str( model["topics_diversity_dist"] / model["count"]) + '\n')
            outf.write(model_name + " top_words_diversity_dist: " + str( model["top_words_diversity_dist"] / model["count"]) + '\n')
            outf.write(model_name + " specificity: "+ str( model["specificity"] / model["count"]) + '\n')
            outf.write(model_name + " novelty: "+ str( model["novelty"] / model["count"]) + '\n')
            outf.write(model_name + " novelty_word: "+ str( model["novelty_word"] / model["count"]) + '\n')
            outf.write(model_name + " novelty_word_w: "+ str( model["novelty_word_w"] / model["count"]) + '\n')
            outf.write(model_name + " relevancy_f_25: "+ str( model["relevancy_f_25"] / model["relevancy_f_25_count"]) + '\n')
            outf.write(model_name + " relevancy_f_25_arr: "+ str( [model["relevancy_f_25_arr"][x] / model["relevancy_f_25_count_arr"][x] for x in range(len(model["relevancy_f_25_count_arr"]))] ) + '\n')
            outf.write(model_name + " context_len_arr: "+ str( [model["context_len_arr"][x] / model["relevancy_f_25_count_arr"][x] for x in range(len(model["relevancy_f_25_count_arr"]))] ) + '\n')
            outf.write(model_name + " context_len_all: "+ str( sum(model["context_len_arr"]) / float(sum(model["relevancy_f_25_count_arr"])) ) + '\n')
            outf.write(model_name + " relevancy_f_25_w: "+ str( model["relevancy_f_25_w"] / model["relevancy_f_25_count_w"]) + '\n')
            outf.write(model_name + " relevancy_topic_f_25: "+ str( model["relevancy_topic_f_25"] / model["relevancy_f_25_count"]) + '\n')
            outf.write(model_name + " relevancy_f_all: "+ str( model["relevancy_f_all"] / model["relevancy_f_all_count"]) + '\n')
            outf.write(model_name + " relevancy_f_25 - (1 - novelty_word): "+ str( (model["relevancy_f_25"] / model["relevancy_f_25_count"] + model["novelty_word"] / model["count"] ) - 1 ) + '\n')
            outf.write(model_name + " relevancy_f_25_w - (1 - novelty_word_w): "+ str( (model["relevancy_f_25_w"] / model["relevancy_f_25_count_w"] + model["novelty_word_w"] / model["count"] ) - 1 ) + '\n')
            outf.write('\n')


