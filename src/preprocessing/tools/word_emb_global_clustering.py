import sys
sys.path.insert(0, sys.path[0]+'/../..')
import utils
from sklearn.cluster import KMeans
import numpy as np

num_special_token = 3

cluster_num = 1000

#emb_file = '/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/resources/glove.840B.300d_filtered_wiki2016_min100.txt'
#dictionary_input_name = './data/processed/wiki2016_gpt2/dictionary_index'
#output_file = '/iesl/canvas/hschang/language_modeling/interactive_LM/models/GloVe_clustering/normalized_center_emb_n'+str(cluster_num)
emb_file = './resources/glove.840B.300d.txt'
dictionary_input_name = './data/processed/wiki2016_gpt2/dictionary_index'
output_file = './models/GloVe_clustering/normalized_center_emb_n'+str(cluster_num)


def load_emb_file_to_np(emb_file, idx2word_freq):
    word2emb, emb_size = utils.load_emb_file_to_dict(emb_file, convert_np = False)
    num_w = len(idx2word_freq)
    #external_emb = np.empty( (num_w, emb_size) )
    external_emb_list = [] 
    OOV_freq = 0
    total_freq = 0
    oov_list = []
    for i in range(num_special_token, num_w):
        w = idx2word_freq[i][0]
        total_freq += idx2word_freq[i][1]
        if w in word2emb:
            val = word2emb[w]
            external_emb_list.append(val)
            #external_emb[i,:] = val
        else:
            oov_list.append(i)
            #external_emb[i,:] = 0
            OOV_freq += idx2word_freq[i][1]
    external_emb = np.array(external_emb_list)
    print("OOV word type percentage: {}%".format( len(oov_list)/float(num_w)*100 ))
    print("OOV token percentage: {}%".format( OOV_freq/float(total_freq)*100 ))
    return external_emb, emb_size, oov_list

with open(dictionary_input_name) as f_in:
    idx2word_freq = utils.load_idx2word_freq(f_in)

external_emb, emb_size, oov_list = load_emb_file_to_np(emb_file, idx2word_freq)
external_emb = external_emb / (0.000000000001 + np.linalg.norm(external_emb,axis = 1, keepdims=True))

kmeans_model = KMeans(n_clusters=cluster_num, n_init=1, random_state=0, init='random', max_iter = 1000)

kmeans_model.fit(external_emb)
np.savetxt(output_file, kmeans_model.cluster_centers_)
