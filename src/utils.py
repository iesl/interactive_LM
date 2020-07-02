import os, shutil
import torch
import torch.utils.data
import torch.nn as nn
import numpy as np
import random
import sys
from gpt2_model.modeling_gpt2 import GPT2Model
from gpt2_model.configuration_gpt2 import GPT2Config

import model as model_code

UNK_IND = 1
EOS_IND = 2

w_d2_ind_init = {'[null]': 0, '<unk>': 1, '<eos>': 2}
ind_l2_w_freq_init = [ ['[null]',-1,0], ['<unk>',0,1], ['<eos>',0,2] ]
num_special_token = len(w_d2_ind_init)

class Dictionary(object):
    def __init__(self, byte_mode=False):
        self.w_d2_ind = w_d2_ind_init
        self.ind_l2_w_freq = ind_l2_w_freq_init
        self.num_special_token = num_special_token
        self.UNK_IND = UNK_IND
        self.EOS_IND = EOS_IND
        self.byte_mode = byte_mode

    def dict_check_add(self,w):
        if w not in self.w_d2_ind:
            w_ind = len(self.w_d2_ind)
            self.w_d2_ind[w] = w_ind
            if self.byte_mode:
                self.ind_l2_w_freq.append([w.decode('utf-8'), 1, w_ind])
            else:
                self.ind_l2_w_freq.append([w, 1, w_ind])
        else:
            w_ind = self.w_d2_ind[w]
            self.ind_l2_w_freq[w_ind][1] += 1
        return w_ind

    def append_eos(self,w_ind_list):
        w_ind_list.append(self.EOS_IND) # append <eos>
        self.ind_l2_w_freq[self.EOS_IND][1] += 1

    def densify_index(self,min_freq):
        vocab_size = len(self.ind_l2_w_freq)
        compact_mapping = [0]*vocab_size
        for i in range(self.num_special_token):
            compact_mapping[i] = i
        #compact_mapping[1] = 1
        #compact_mapping[2] = 2

        #total_num_filtering = 0
        total_freq_filtering = 0
        current_new_idx = self.num_special_token

        #for i, (w, w_freq, w_ind_org) in enumerate(self.ind_l2_w_freq[self.num_special_token:]):
        for i in range(self.num_special_token,vocab_size):
            w, w_freq, w_ind_org = self.ind_l2_w_freq[i]
            if w_freq < min_freq:
                compact_mapping[i] = self.UNK_IND
                self.ind_l2_w_freq[i][-1] = self.UNK_IND
                self.ind_l2_w_freq[i].append('unk')
                #total_num_filtering += 1
                total_freq_filtering += w_freq
            else:
                compact_mapping[i] = current_new_idx
                self.ind_l2_w_freq[i][-1] = current_new_idx
                current_new_idx += 1

        self.ind_l2_w_freq[self.UNK_IND][1] = total_freq_filtering #update <unk> frequency


        print("{}/{} word types are filtered".format(vocab_size - current_new_idx, vocab_size) )

        return compact_mapping, total_freq_filtering

    def store_dict(self,f_out):
        vocab_size = len(self.ind_l2_w_freq)
        for i in range(vocab_size):
            #print(ind_l2_w_freq[i])
            self.ind_l2_w_freq[i][1] = str(self.ind_l2_w_freq[i][1])
            self.ind_l2_w_freq[i][2] = str(self.ind_l2_w_freq[i][2])
            f_out.write('\t'.join(self.ind_l2_w_freq[i])+'\n')

class Condition_Seq2PairDataset(torch.utils.data.Dataset):
#will need to handle the partial data loading if the dataset size is larger than cpu memory
    def __init__(self, w_ind_gpt2_tensor, w_ind_spacy_tensor, idx_gpt2_to_spacy_tensor, bptt, n_further, dilated_head_span_no_use, random_start_no_use, device):
        self.w_ind_gpt2 = w_ind_gpt2_tensor
        self.w_ind_spacy = w_ind_spacy_tensor
        self.idx_gpt2_to_spacy = idx_gpt2_to_spacy_tensor
        self.n_further = n_further
        self.seq_len = bptt
        #self.dilated_head_span = dilated_head_span
        #self.head_num = int(bptt / dilated_head_span) + 2
        self.output_device = device

    def __len__(self):
        return int( (self.w_ind_gpt2.size(0) - self.seq_len)/self.seq_len )

    def __getitem__(self, idx):
        feature = self.w_ind_gpt2[idx*self.seq_len:(idx+1)*self.seq_len].to(dtype = torch.long, device = self.output_device)
        idx_gpt2_to_spacy_small = self.idx_gpt2_to_spacy[idx*self.seq_len:(idx+1)*self.seq_len].to(dtype = torch.long, device = self.output_device)
        idx_gpt2_to_spacy_small = idx_gpt2_to_spacy_small - idx_gpt2_to_spacy_small[0]
        #init_head_posi = random.randint(0, self.dilated_head_span - 1)
        #inner_idx_tensor = torch.empty(self.head_num, dtype = torch.long, device = self.output_device)
        #future_mask = torch.zeros( (self.head_num, self.seq_len), dtype = torch.float, device = self.output_device)
        #for i in range(self.head_num):
        #    inner_idx = min(self.seq_len-1, i*self.dilated_head_span+init_head_posi)
        #    inner_idx_tensor[i] = inner_idx
        #    future_mask[i,:inner_idx+1] = 1
        
        if self.idx_gpt2_to_spacy is None or self.n_further<0:
            target_unfold = []
        else:
            #spacy_idx = torch.empty( (self.head_num, self.n_further), dtype = torch.long, device = self.output_device)
            #for i in range(self.head_num):
            #    inner_idx = inner_idx_tensor[i]
            #    spacy_idx[i,:] = torch.tensor(list(range( self.idx_gpt2_to_spacy[idx*self.seq_len+inner_idx] - self.idx_gpt2_to_spacy[idx*self.seq_len], self.idx_gpt2_to_spacy[idx*self.seq_len+inner_idx] - self.idx_gpt2_to_spacy[idx*self.seq_len] + self.n_further )), dtype = torch.long, device = self.output_device)

            start = self.idx_gpt2_to_spacy[idx*self.seq_len] + 2
            end = min(len(self.w_ind_spacy), self.idx_gpt2_to_spacy[(idx+1)*self.seq_len] + 2 + self.n_further)
            target_small = self.w_ind_spacy[start:end]
            #handle the out of boundary case
            #oob_num = end - self.w_ind_spacy.size(0)
            #if oob_num >= 0:
            #    target_small = torch.cat( (self.w_ind_spacy[start:],torch.zeros(oob_num+1, dtype = torch.int) ), dim=0 ).to( dtype = torch.long, device = self.output_device)
            #else:
            #    target_small = self.w_ind_spacy[start:end].to( dtype = torch.long, device = self.output_device)
            target_size = target_small.size(0)
            add_0_num = self.seq_len + self.n_further - target_size + 1000
            if add_0_num > 0:
                target_small = torch.cat( (target_small,torch.zeros(add_0_num, dtype = torch.int) ), dim=0 ).to( dtype = torch.long, device = self.output_device)
            else:
                target_small = target_small.to( dtype = torch.long, device = self.output_device)


            #target_unfold = target_small[spacy_idx]
            #try:
            #    target_unfold = target_small[spacy_idx]
            #except:
            #    print(target_small, spacy_idx, oob_num, start, end, idx, self.idx_gpt2_to_spacy[idx*self.seq_len])

        #return [feature, target_unfold, inner_idx_tensor, future_mask]
        return [feature, target_small, idx_gpt2_to_spacy_small]


class Seq2PairDataset(torch.utils.data.Dataset):
#will need to handle the partial data loading if the dataset size is larger than cpu memory
    def __init__(self, w_ind_gpt2_tensor, w_ind_spacy_tensor, idx_gpt2_to_spacy_tensor, bptt, n_further, dilated_head_span, random_start, device):
        self.w_ind_gpt2 = w_ind_gpt2_tensor
        self.w_ind_spacy = w_ind_spacy_tensor
        self.idx_gpt2_to_spacy = idx_gpt2_to_spacy_tensor
        self.n_further = n_further
        self.seq_len = bptt
        self.dilated_head_span = dilated_head_span
        self.head_num = int(bptt / dilated_head_span) + 2
        self.random_start = random_start
        self.output_device = device

    def __len__(self):
        return int( (self.w_ind_gpt2.size(0) - self.seq_len)/self.seq_len )

    def __getitem__(self, idx):
        feature = self.w_ind_gpt2[idx*self.seq_len:(idx+1)*self.seq_len].to(dtype = torch.long, device = self.output_device)
        if self.random_start:
            init_head_posi = random.randint(1, self.dilated_head_span - 1)
        else:
            #init_head_posi = 20
            init_head_posi = int( self.dilated_head_span / 2 )
        inner_idx_tensor = torch.empty(self.head_num, dtype = torch.long, device = self.output_device)
        future_mask = torch.zeros( (self.head_num, self.seq_len), dtype = torch.float, device = self.output_device)
        for i in range(self.head_num):
            inner_idx = min(self.seq_len-1, i*self.dilated_head_span+init_head_posi)
            inner_idx_tensor[i] = inner_idx
            future_mask[i,:inner_idx+1] = 1
        
        if self.idx_gpt2_to_spacy is None or self.n_further<0:
            target_unfold = []
            #idx_feature_to_target = []
            #inner_idx_tensor = []
            #future_mask = []
        else:
            spacy_idx = torch.empty( (self.head_num, self.n_further), dtype = torch.long, device = self.output_device)
            #spacy_idx = torch.empty( (self.head_num, self.n_further), dtype = torch.long)
            #inner_idx_arr = []
            #future_mask = torch.ones( (self.head_num, self.seq_len), dtype = torch.bool, device = self.output_device)
            for i in range(self.head_num):
                inner_idx = inner_idx_tensor[i]
                spacy_idx[i,:] = torch.tensor(list(range( self.idx_gpt2_to_spacy[idx*self.seq_len+inner_idx] - self.idx_gpt2_to_spacy[idx*self.seq_len], self.idx_gpt2_to_spacy[idx*self.seq_len+inner_idx] - self.idx_gpt2_to_spacy[idx*self.seq_len] + self.n_further )), dtype = torch.long, device = self.output_device)

            start = self.idx_gpt2_to_spacy[idx*self.seq_len] + 1
            end = self.idx_gpt2_to_spacy[(idx+1)*self.seq_len] + 1 + self.n_further
            #handle the out of boundary case
            oob_num = end - self.w_ind_spacy.size(0)
            if oob_num >= 0:
                target_small = torch.cat( (self.w_ind_spacy[start:],torch.zeros(oob_num+1, dtype = torch.int) ), dim=0 ).to( dtype = torch.long, device = self.output_device)
            else:
                target_small = self.w_ind_spacy[start:end].to( dtype = torch.long, device = self.output_device)
            #print(target_small, spacy_idx)
            target_unfold = target_small[spacy_idx]
            #try:
            #    target_unfold = target_small[spacy_idx]
            #except:
            #    print(target_small, spacy_idx, oob_num, start, end, idx, self.idx_gpt2_to_spacy[idx*self.seq_len])
            #idx_feature_to_target = self.idx_gpt2_to_spacy[idx:idx+self.seq_len].to(dtype = torch.long, device = self.output_device)
            #idx_feature_to_target -= idx_feature_to_target[0]

        #debug target[-1] = idx
        return [feature, target_unfold, inner_idx_tensor, future_mask]

#def get_batch_further(source, i, n_further, seq_len):
#    n_further_m1 = n_further - 1
#    target_further = torch.zeros([ seq_len + n_further_m1, source.size(1) ], dtype=torch.long, requires_grad = False)
#    further_len = min(n_further_m1, len(source) - i - 2 - seq_len)
#    #if further_len < 0:
#    #    print("the final batch size might be too small")
#    target_further[:seq_len + further_len] = source[i+2:i+2+seq_len+further_len]
#    return target_further

def load_idx2word_freq(f_in):
    idx2word_freq = []
    for i, line in enumerate(f_in):
        fields = line.rstrip().split('\t')
        if len(fields) == 3:
            assert len(idx2word_freq) == int(fields[2])
            idx2word_freq.append([fields[0],int(fields[1])])            

    return idx2word_freq

def create_data_loader_split(f_in, bsz, bptt, n_further, dilated_head_span, device, split_num, dataset_class):
    w_ind_gpt2_tensor, w_ind_spacy_tensor, idx_gpt2_to_spacy_tensor = torch.load(f_in, map_location='cpu')
    #print(w_ind_gpt2_tensor.size(), w_ind_spacy_tensor.size(), idx_gpt2_to_spacy_tensor.size())
    training_size = w_ind_gpt2_tensor.size(0)
    idx_arr = np.random.permutation(training_size)
    split_size = int(training_size / split_num)
    dataset_arr = []
    for i in range(split_num):
        start = i * split_size
        if i == split_num - 1:
            end = training_size
        else:
            end = (i+1) * split_size
        start_idx_spacy = idx_gpt2_to_spacy_tensor[start]
        end_idx_spacy = idx_gpt2_to_spacy_tensor[end-1]
        random_start = True
        dataset_arr.append(  dataset_class(w_ind_gpt2_tensor[start:end],w_ind_spacy_tensor[start_idx_spacy:end_idx_spacy], idx_gpt2_to_spacy_tensor[start:end]-start_idx_spacy, bptt, n_further, dilated_head_span, random_start, device ) ) #assume that the dataset are randomly shuffled beforehand

    use_cuda = False
    if device.type == 'cude':
        use_cuda = True
    #dataloader_arr = [torch.utils.data.DataLoader(dataset_arr[i], batch_size = bsz, shuffle = True, pin_memory=use_cuda, drop_last=False) for i in range(split_num)]
    dataloader_arr = [torch.utils.data.DataLoader(dataset_arr[i], batch_size = bsz, shuffle = True, pin_memory=use_cuda, drop_last=True) for i in range(split_num)]
    return dataloader_arr

def create_data_loader(f_in, bsz, bptt, n_further, dilated_head_span, device, dataset_class, want_to_shuffle = True, random_start=True):
    w_ind_gpt2_tensor, w_ind_spacy_tensor, idx_gpt2_to_spacy_tensor = torch.load(f_in, map_location='cpu')
    dataset = dataset_class(w_ind_gpt2_tensor, w_ind_spacy_tensor, idx_gpt2_to_spacy_tensor, bptt, n_further, dilated_head_span, random_start, device)
    use_cuda = False
    if device.type == 'cude':
        use_cuda = True
    #return torch.utils.data.DataLoader(dataset, batch_size = bsz, shuffle = want_to_shuffle, pin_memory=use_cuda, drop_last=False)
    return torch.utils.data.DataLoader(dataset, batch_size = bsz, shuffle = want_to_shuffle, pin_memory=use_cuda, drop_last=True)


def load_corpus(data_path, train_bsz, eval_bsz, bptt, n_further, dilated_head_span, device, tensor_folder = "tensors", split_num = 1, skip_training = False, want_to_shuffle_val = False, condition = False, load_testing = False, random_start = True):
    train_corpus_name = data_path + "/" + tensor_folder + "/train.pt"
    val_org_corpus_name = data_path +"/" + tensor_folder + "/val_org.pt"
    dictionary_input_name = data_path + "/" + tensor_folder + "/dict_idx_compact"
    
    if condition:
        dataset_class = Condition_Seq2PairDataset
    else:
        dataset_class = Seq2PairDataset

    with open(dictionary_input_name) as f_in:
        idx2word_freq = load_idx2word_freq(f_in)

    with open(val_org_corpus_name,'rb') as f_in:
        dataloader_val = create_data_loader(f_in, eval_bsz, bptt, n_further, dilated_head_span, device, dataset_class, want_to_shuffle = want_to_shuffle_val, random_start=random_start)
    
    if load_testing:
        test_org_corpus_name = data_path +"/" + tensor_folder + "/test_org.pt"
        with open(test_org_corpus_name,'rb') as f_in:
            dataloader_test = create_data_loader(f_in, eval_bsz, bptt, n_further, dilated_head_span, device, dataset_class, want_to_shuffle = want_to_shuffle_val, random_start=random_start)
    
    if skip_training:
        dataloader_train_arr = [0]
    else:
        with open(train_corpus_name,'rb') as f_in:
            dataloader_train_arr = create_data_loader_split(f_in, train_bsz, bptt, n_further, dilated_head_span, device, split_num, dataset_class)

    if load_testing:
        return idx2word_freq, dataloader_train_arr, dataloader_val, dataloader_test
    else:
        return idx2word_freq, dataloader_train_arr, dataloader_val

def load_emb_from_path(emb_file_path, device, idx2word_freq):
    if emb_file_path[-3:] == '.pt':
        word_emb = torch.load( emb_file_path, map_location=device )
        output_emb_size = word_emb.size(1)
    else:
        word_emb, output_emb_size, oov_list = load_emb_file_tensor(emb_file_path,device,idx2word_freq)
    return word_emb, output_emb_size

def loading_all_models(args, idx2word_freq, device):

    word_emb, output_emb_size = load_emb_from_path(args.emb_file, device, idx2word_freq)

    model_name = 'distilgpt2'
    
    encoder_state_dict = torch.load(os.path.join(args.checkpoint_topics, 'encoder.pt'), map_location=device)
    encoder = GPT2Model.from_pretrained(model_name, state_dict = encoder_state_dict)
    gpt2_config = GPT2Config.from_pretrained(model_name)
    #encoder = model_code.SEQ2EMB(args.en_model.split('+'), ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouti, args.dropoute, max_sent_len,  external_emb, [], trans_layers = args.encode_trans_layers, trans_nhid = args.trans_nhid) 

    if args.nhidlast2 < 0:
        args.nhidlast2 = encoder.output_dim
    decoder = model_code.EMB2SEQ(args.de_model.split('+'), gpt2_config.n_embd, args.nhidlast2, output_emb_size, 1, args.n_basis, positional_option = args.positional_option, dropoutp= args.dropoutp, trans_layers = args.trans_layers, using_memory = args.de_en_connection, dropout_prob_trans = args.dropout_prob_trans) 

    #encoder.load_state_dict(torch.load(os.path.join(args.checkpoint, 'encoder.pt'), map_location=device))
    decoder.load_state_dict(torch.load(os.path.join(args.checkpoint_topics, 'decoder.pt'), map_location=device))

    #if len(args.emb_file) == 0:
    #    word_emb = encoder.encoder.weight.detach()

    word_norm_emb = word_emb / (0.000000000001 + word_emb.norm(dim = 1, keepdim=True) )
    word_norm_emb[0,:] = 0

    parallel_encoder, parallel_decoder = output_parallel_models(args.cuda_topics, args.single_gpu, encoder, decoder)

    return parallel_encoder, parallel_decoder, encoder, decoder, word_norm_emb

def seed_all_randomness(seed,use_cuda):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        if not use_cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(seed)

def output_parallel_models(use_cuda, single_gpu, encoder, decoder):
    if use_cuda:
        if single_gpu:
            parallel_encoder = encoder.cuda()
            parallel_decoder = decoder.cuda()
        else:
            parallel_encoder = nn.DataParallel(encoder, dim=0).cuda()
            parallel_decoder = nn.DataParallel(decoder, dim=0).cuda()
            #parallel_decoder = decoder.cuda()
    else:
        parallel_encoder = encoder
        parallel_decoder = decoder
    return parallel_encoder, parallel_decoder

def load_emb_file_to_dict(emb_file, lowercase_emb = False, convert_np = True):
    word2emb = {}
    with open(emb_file) as f_in:
        for line in f_in:
            word_val = line.rstrip().split(' ')
            if len(word_val) < 3:
                continue
            word = word_val[0]
            #val = np.array([float(x) for x in  word_val[1:]])
            val = [float(x) for x in  word_val[1:]]
            if convert_np:
                val = np.array(val)
            if lowercase_emb:
                word_lower = word.lower()
                if word_lower not in word2emb:
                    word2emb[word_lower] = val
                else:
                    if word == word_lower:
                        word2emb[word_lower] = val
            else:
                word2emb[word] = val
            emb_size = len(val)
    return word2emb, emb_size

def load_emb_file_to_tensor(emb_file, device, idx2word_freq):
    word2emb, emb_size = load_emb_file_to_dict(emb_file, convert_np = False)
    num_w = len(idx2word_freq)
    #emb_size = len(word2emb.values()[0])
    #external_emb = torch.empty(num_w, emb_size, device = device, requires_grad = update_target_emb)
    external_emb = torch.empty(num_w, emb_size, device = device, requires_grad = False)
    #OOV_num = 0
    OOV_freq = 0
    total_freq = 0
    oov_list = []
    for i in range(num_special_token, num_w):
        w = idx2word_freq[i][0]
        total_freq += idx2word_freq[i][1]
        if w in word2emb:
            val = torch.tensor(word2emb[w], device = device, requires_grad = False)
            #val = np.array(word2emb[w])
            external_emb[i,:] = val
        else:
            oov_list.append(i)
            external_emb[i,:] = 0
            #OOV_num += 1
            OOV_freq += idx2word_freq[i][1]

    print("OOV word type percentage: {}%".format( len(oov_list)/float(num_w)*100 ))
    print("OOV token percentage: {}%".format( OOV_freq/float(total_freq)*100 ))
    return external_emb, emb_size, oov_list

def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)

    print('Experiment dir : {}'.format(path))
    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

def str2bool(v):
    if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def save_checkpoint(encoder, decoder, optimizer_e,  optimizer_d, external_emb, path):
    torch.save(encoder.state_dict(), os.path.join(path, 'encoder.pt'))
    try:
        torch.save(decoder.state_dict(), os.path.join(path, 'decoder.pt'))
    except:
        pass
    if external_emb.size(0) > 1:
        torch.save(external_emb, os.path.join(path, 'target_emb.pt'))
    torch.save(optimizer_e.state_dict(), os.path.join(path, 'optimizer_e.pt'))
    torch.save(optimizer_d.state_dict(), os.path.join(path, 'optimizer_d.pt'))
