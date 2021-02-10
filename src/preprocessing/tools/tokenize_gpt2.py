import getopt
import sys
sys.path.insert(0, sys.path[0]+'/../..')
#from transformers import GPT2Tokenizer
from gpt2_model.tokenization_gpt2 import GPT2Tokenizer

#help_msg = '-i <input_file_path> -t <tokenize_sents> -o <output_file_path>'
help_msg = '-i <input_file_path> -o <output_file_path>'

#tokenize_sents = 1

try:
    #opts, args = getopt.getopt(sys.argv[1:], "i:t:o:")
    opts, args = getopt.getopt(sys.argv[1:], "i:o:")
except getopt.GetoptError:
    print(help_msg)
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print(help_msg)
        sys.exit()
    elif opt in ("-i"):
        input_file_name = arg
    #elif opt in ("-t"):
    #    tokenize_sents = int(arg)
    elif opt in ("-o"):
        output_file_name = arg

GPT2_model_name = 'distilgpt2'

#unused_sent = ''

tokenizer = GPT2Tokenizer.from_pretrained(GPT2_model_name)

f_out = open(output_file_name,'w')

with open(input_file_name) as f_in:
    print("Processing "+ input_file_name)
    for line_idx, line in enumerate(f_in):
        if line_idx % 10000 == 0:
            sys.stdout.write(str(line_idx)+' ')
            sys.stdout.flush()
        fields = line.rstrip().split('\t')
        #if len(fields) == 1:
        #    unused_sent += fields[0] + ' '
        #    continue
        if len(fields)<2:
            continue
        if len(fields)>2:
            print(fields)
        #    continue
        org_sent, spacy_sent = fields
        #if len(unused_sent) > 0:
        #    org_sent = unused_sent + ' ' + org_sent
        #    if spacy_sent[0] != org_sent[0]:
        #        unused_sent = org_sent
        #        continue
        #    unused_sent = ''
        #print(spacy_sent)
        tokenized_text = tokenizer.tokenize(org_sent, add_prefix_space=True)
        #print(tokenized_text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        #print(indexed_tokens)
        spacy_token_idx = []
        spacy_character = []
        for i, token in enumerate(spacy_sent.split()):
            spacy_character+=[x for x in token]
            spacy_token_idx+=[i]*len(token)
        gpt_token_in_spacy = []
        current_idx = 0
        reaching_end = False
        for token_gpt in tokenized_text:
            #print(current_idx)
            if token_gpt[0] == "Ġ" or token_gpt[0] == "ġ":
                token_gpt = token_gpt[1:]
            #print(token_gpt)
            if current_idx+len(token_gpt) >= len(spacy_character) or reaching_end:
                reaching_end = True
                gpt_token_in_spacy.append(spacy_token_idx[-1])
                continue
            if len(token_gpt) == 0:
                gpt_token_in_spacy.append(spacy_token_idx[max(0,current_idx-1)])
                continue
            if token_gpt != spacy_character[current_idx:current_idx+len(token_gpt)]:
                #print(tokenized_text, token_gpt, current_idx, spacy_character)
                while(current_idx < len(spacy_character) and ord(spacy_character[current_idx])>=128 ):
                    current_idx += 1
                if ord(token_gpt[0]) >= 128:
                    gpt_token_in_spacy.append(spacy_token_idx[max(0,current_idx-1)])
                    continue
            #assert token_gpt[0] == spacy_character[current_idx], print(tokenized_text, token_gpt, current_idx, spacy_character)
            current_idx += len(token_gpt)
            #assert current_idx - 1 < len(spacy_token_idx), print(spacy_sent, tokenized_text, token_gpt, current_idx, spacy_character)
            assert current_idx - 1 < len(spacy_token_idx), print(input_file_name)
            gpt_token_in_spacy.append(spacy_token_idx[current_idx-1])
        #assert len(gpt_token_in_spacy) == len(indexed_tokens)
        for i in range(len(gpt_token_in_spacy)-1):
            assert gpt_token_in_spacy[i+1] >= gpt_token_in_spacy[i], print(input_file_name, gpt_token_in_spacy, spacy_sent, spacy_token_idx, tokenized_text)
        f_out.write(spacy_sent+'\t'+ ' '.join([str(x) for x in indexed_tokens]) + '\t' +' '.join([str(x) for x in gpt_token_in_spacy])+'\n')
        #print(gpt_token_in_spacy)
        #print(tokenized_text)
f_out.close()
