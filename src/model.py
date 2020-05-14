import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import model_trans
import sys
#from weight_drop import WeightDrop
class SparseCoding(nn.Module):
    def __init__(self, ntopic, nbow, emb_size, device):
        super(SparseCoding, self).__init__()
        self.code_book = nn.Parameter(torch.randn(ntopic, emb_size, device=device, requires_grad=True))
        self.coeff = nn.Parameter(torch.randn(nbow, ntopic,  device=device, requires_grad=True))
        self.device = device
    
    def compute_coeff_pos(self):
        self.coeff.data = self.coeff.clamp(0.0, 1.0)
        #self.coeff.data = self.coeff.clamp(0.0)
    
    def forward(self):
        result = self.coeff.matmul(self.code_book)
        return result


class MatrixReconstruction(nn.Module):
    def __init__(self, batch_size, ntopic, nbow, device):
        super(MatrixReconstruction, self).__init__()
        self.coeff = nn.Parameter(torch.randn(batch_size, ntopic, nbow, device=device, requires_grad=True))
        self.device = device
    
    def compute_coeff_pos(self):
        self.coeff.data = self.coeff.clamp(0.0, 1.0)
    
    def forward(self, input):
        result = self.coeff.matmul(input)
        return result

class RNN_decoder(nn.Module):
    def __init__(self, model_type, emb_dim, ninp, nhid, nlayers):
        super(RNN_decoder, self).__init__()
        if model_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, model_type)(emb_dim, nhid, nlayers, dropout=0)
            #if linear_mapping_dim > 0:
            #    self.rnn = getattr(nn, model_type)(linear_mapping_dim, nhid, nlayers, dropout=0)
            #else:
            #    self.rnn = getattr(nn, model_type)(ninp+position_emb_size, nhid, nlayers, dropout=0)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[model_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            #self.rnn = nn.RNN(ninp+position_emb_size, nhid, nlayers, nonlinearity=nonlinearity, dropout=0)
            self.rnn = nn.RNN(emb_dim, nhid, nlayers, nonlinearity=nonlinearity, dropout=0)
        
        if model_type == 'LSTM':
            self.init_hid_linear_1 = nn.ModuleList([nn.Linear(ninp, nhid) for i in range(nlayers)])
            self.init_hid_linear_2 = nn.ModuleList([nn.Linear(ninp, nhid) for i in range(nlayers)])
            for i in range(nlayers):
                self.init_hid_linear_1[i].weight.data.uniform_(-.1,.1)
                self.init_hid_linear_1[i].bias.data.uniform_(-.5,.5)
                self.init_hid_linear_2[i].weight.data.uniform_(-.1,.1)
                self.init_hid_linear_2[i].bias.data.uniform_(-.5,.5)
        self.nlayers = nlayers
        self.model_type = model_type

    def forward(self, input_init, emb):
        hidden_1 = torch.cat( [self.init_hid_linear_1[i](input_init).unsqueeze(dim = 0) for i in range(self.nlayers)], dim = 0 )
        hidden_2 = torch.cat( [self.init_hid_linear_2[i](input_init).unsqueeze(dim = 0) for i in range(self.nlayers)], dim = 0 )
        hidden = (hidden_1, hidden_2)
        
        output, hidden = self.rnn(emb, hidden)
        return output

    #def init_hidden(self, bsz):
    #    weight = next(self.parameters())
    #    if self.model_type == 'LSTM':
    #        return (weight.new_zeros(self.nlayers, bsz, self.nhid),
    #                weight.new_zeros(self.nlayers, bsz, self.nhid))
    #    else:
    #        return weight.new_zeros(self.nlayers, bsz, self.nhid)
    

class ext_emb_to_seq(nn.Module):
    def __init__(self, model_type_list, emb_dim, ninp, nhid, nlayers, n_basis, trans_layers, using_memory, add_position_emb, dropout_prob_trans):
        super(ext_emb_to_seq, self).__init__()
        self.decoder_array = nn.ModuleList()
        input_dim = emb_dim
        self.trans_dim = None
        for model_type in model_type_list:
            if model_type == 'LSTM':
                model = RNN_decoder(model_type, input_dim, ninp, nhid, nlayers)
                input_dim = nhid
                #output_dim = nhid
            elif model_type == 'TRANS':
                #model = model_trans.BertEncoder(model_type = model_type, hidden_size = input_dim, max_position_embeddings = n_basis, num_hidden_layers=trans_layers)
                #print("input_dim", input_dim)
                model = model_trans.Transformer(model_type = model_type, hidden_size = input_dim, max_position_embeddings = n_basis, num_hidden_layers=trans_layers, add_position_emb = add_position_emb,  decoder = using_memory, dropout_prob = dropout_prob_trans)
                self.trans_dim = input_dim
                #output_dim = input_dim
            else:
                print("model type must be either LSTM or TRANS")
                sys.exit(1)
            self.decoder_array.append( model )
        self.output_dim = input_dim

    def forward(self, input_init, emb, memory=None, memory_attention_mask=None):
        hidden_states = emb
        for model in self.decoder_array:
            model_type = model.model_type
            if model_type == 'LSTM':
                hidden_states = model(input_init, hidden_states)
            elif model_type == 'TRANS':
                #If we want to use transformer by default at the end, we will want to reconsider reducing the number of permutes
                hidden_states = hidden_states.permute(1,0,2)
                hidden_states = model(hidden_states, memory, memory_attention_mask)
                hidden_states = hidden_states[0].permute(1,0,2)
        return hidden_states

#class RNNModel_decoder(nn.Module):
class EMB2SEQ(nn.Module):

    #def __init__(self, model_type_list, ninp, nhid, outd, nlayers, n_basis, linear_mapping_dim, dropoutp= 0.5, trans_layers=2, using_memory = False):
    def __init__(self, model_type_list, ninp, nhid, outd, nlayers, n_basis, positional_option, dropoutp= 0.5, trans_layers=2, using_memory = False, dropout_prob_trans = 0.1):
        #super(RNNModel_decoder, self).__init__()
        super(EMB2SEQ, self).__init__()
        self.drop = nn.Dropout(dropoutp)
        self.n_basis = n_basis
        #self.layernorm = nn.InstanceNorm1d(n_basis, affine =False)
        #self.outd_sqrt = math.sqrt(outd)
        #self.linear_mapping_dim = linear_mapping_dim
        #position_emb_size = 0
        #if n_basis > 0:
        #if linear_mapping_dim > 0:
        input_size = ninp
        #self.in_linear = nn.Linear(ninp, nhid)
        if using_memory:
            self.in_linear_mem = nn.Linear(ninp, nhid)
            self.in_linear_mem.bias.data.uniform_(-.00005,.00005)
            self.in_linear_mem.weight.data.uniform_(-.00001,.00001)
        #input_size = nhid
        add_position_emb = False
        if positional_option == 'linear':
            linear_mapping_dim = nhid
            self.init_linear_arr = nn.ModuleList([nn.Linear(input_size, linear_mapping_dim) for i in range(n_basis)])
            for i in range(n_basis):
                #It seems that the LSTM only learns well when bias is larger than weights at the beginning
                #If setting std in weight to be too large (e.g., 1), the loss might explode
                self.init_linear_arr[i].bias.data.uniform_(-.5,.5)
                self.init_linear_arr[i].weight.data.uniform_(-.1,.1)
            input_size = linear_mapping_dim
        elif positional_option == 'cat':
            position_emb_size = 100
            self.poistion_emb = nn.Embedding( n_basis, position_emb_size )
            self.linear_keep_same_dim = nn.Linear(position_emb_size + input_size, nhid)
            #input_size = position_emb_size + ninp
            #input_size = ninp
        elif positional_option == 'add':
            #input_size = ninp
            add_position_emb = True
            if model_type_list[0] == 'LSTM':
                self.poistion_emb = nn.Embedding( n_basis, input_size )
            else:
                self.scale_factor = math.sqrt(input_size)
                
        #self.relu_layer = nn.ReLU()
        
        self.positional_option = positional_option
        self.dep_learner = ext_emb_to_seq(model_type_list, nhid, input_size, nhid, nlayers, n_basis, trans_layers, using_memory, add_position_emb, dropout_prob_trans)
        
        self.trans_dim = self.dep_learner.trans_dim
        

        #self.out_linear = nn.Linear(nhid, outd, bias=False)
        self.out_linear = nn.Linear(self.dep_learner.output_dim, outd)
        
        self.init_weights()

        #self.model_type = model_type
        #self.nhid = nhid

    def init_weights(self):
        #necessary?
        initrange = 0.1
        self.out_linear.bias.data.zero_()
        self.out_linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, input_init, memory = None, memory_attention_mask = None):
        #print(input_init.size())
        def prepare_posi_emb(input, poistion_emb, drop):
            batch_size = input.size(1)
            n_basis = input.size(0)
            input_pos = torch.arange(n_basis,dtype=torch.long,device = input.get_device()).expand(batch_size,n_basis).permute(1,0)
            poistion_emb_input = poistion_emb(input_pos)
            poistion_emb_input = drop(poistion_emb_input)
            return poistion_emb_input

        #input_init_small = in_linear(input_init)
        #input = input_init_small.expand(self.n_basis, input_init_small.size(0), input_init_small.size(1) )
        input = input_init.expand(self.n_basis, input_init.size(0), input_init.size(1) )
        #if self.n_basis == 0:
        #    emb = input
        #else:
        #    if self.linear_mapping_dim > 0:
        if self.positional_option == 'linear':
            emb_raw = torch.cat( [self.init_linear_arr[i](input_init).unsqueeze(dim = 0)  for i in range(self.n_basis) ] , dim = 0 )
            #emb = emb_raw
            emb = self.drop(emb_raw)
        elif self.positional_option == 'cat':
            #batch_size = input.size(1)
            #input_pos = torch.arange(self.n_basis,dtype=torch.long,device = input.get_device()).expand(batch_size,self.n_basis).permute(1,0)

            #poistion_emb_input = self.poistion_emb(input_pos)
            #poistion_emb_input = self.drop(poistion_emb_input)
            poistion_emb_input = prepare_posi_emb(input, self.poistion_emb, self.drop)
            emb = torch.cat( ( poistion_emb_input,input), dim = 2  )
            emb = self.linear_keep_same_dim(emb)
        elif self.positional_option == 'add':
            if self.dep_learner.decoder_array[0].model_type == "LSTM":
                poistion_emb_input = prepare_posi_emb(input, self.poistion_emb, self.drop)
                emb = input + poistion_emb_input
            else:
                emb = input * self.scale_factor
        
        if memory is not None:
            memory = self.in_linear_mem(memory)
        output = self.dep_learner(input_init, emb, memory, memory_attention_mask)
        #output = self.drop(output)
        #output = self.out_linear(self.relu_layer(output))
        #output = self.layernorm(output.permute(1,0,2)).permute(1,0,2)
        #output /= self.outd_sqrt
        output = self.out_linear(output)
        #output = output / (0.000000000001 + output.norm(dim = 2, keepdim=True) )
        output_batch_first = output.permute(1,0,2)

        return output_batch_first


