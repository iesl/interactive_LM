from __future__ import unicode_literals, print_function
#from spacy.en import English
import spacy
#from spacy.pipeline import Sentencizer
from spacy.lang.en import English
import gzip
import json
import sys
import getopt

help_msg = '-i <input_file_path> -t <tokenize_sents> -o <output_file_path>'

tokenize_sents = 1

try:
    opts, args = getopt.getopt(sys.argv[1:], "i:t:o:")
except getopt.GetoptError:
    print(help_msg)
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print(help_msg)
        sys.exit()
    elif opt in ("-i"):
        input_file_name = arg
    elif opt in ("-t"):
        tokenize_sents = int(arg)
    elif opt in ("-o"):
        output_file_name = arg

#input_file_name = "/iesl/data/nmonath-wiki-json/json/enwik0.json.gz"
#output_file_name = "/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/data/raw/wiki2016/enwik0"
#input_file_name = "/iesl/data/nmonath-wiki-json/json/enwik163.json.gz"
#output_file_name = "/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/data/raw/wiki2016/enwik163"

pipeline = ["sentencizer"]
#pipeline = ["parser"]
nlp = English()

#nlp = spacy.load('en', disable = ['tagger', 'ner'])
#nlp.pipeline = [nlp.Sentencizer]
for name in pipeline:
    component = nlp.create_pipe(name)
    nlp.add_pipe(component)

#def dump

output_corpus = []
if input_file_name[-3:] == '.gz':
    f_in = gzip.open(input_file_name)
else:
    f_in = open(input_file_name)
#with gzip.open(input_file_name) as f_in:
print("Processing "+ input_file_name)
text_buffer = []
for line in f_in:
    input_json = json.loads(line)
    text_buffer.append(input_json['text'].replace("'''",'').replace("''",'').replace('"',''))
    #input_json = json.loads(line.decode('utf-8'))


#input_json =json.loads(f_in)
#print(len(input_json))
#print(input_json[1].keys())
#print("Pipeline", nlp.pipe_names)
print("Num of loaded pages", len(text_buffer))
sys.stdout.flush()
#for doc in nlp.pipe(text_buffer, batch_size=10000, n_threads=10, disable=['parser', 'tagger', 'ner']):
#for doc in nlp.pipe(text_buffer, batch_size=10000, n_threads=10):
for i_text, text in enumerate(text_buffer):
    #print(text)
    doc = nlp(text, disable=['parser', 'tagger', 'ner'])
    doc.is_parsed = True
    if i_text%1000 == 0:
        sys.stdout.write(str(i_text)+' ')
        sys.stdout.flush()

    #doc = nlp(input_json['text'], disable=['parser', 'tagger', 'ner'])

    if tokenize_sents == 1:
        for sent in doc.sents:
            w_list = []
            for w in sent:
                if '\n' not in w.text and ' ' not in w.text:
                    w_list.append(w.text)
                #print(w.text)
            if len(w_list) > 0:
                output_corpus.append(w_list)
    elif tokenize_sents == 0:
        for sent in doc.sents:
            output_corpus.append(sent.text.split())
    elif tokenize_sents == 2:
        for sent in doc.sents:
            w_list = []
            for w in sent:
                #if '\n' not in w.text and ' ' not in w.text and '\t' not in w.text:
                w_list.append(w.text)
                #print(w.text)
            output_corpus.append( [sent.text.replace('\n',' ').replace('\t',' '), ' '.join(w_list).replace('\n',' ').replace('\t',' ')] )
    output_corpus.append([])
        #print('eos')
    #if i_text > 100:
    #    break
f_in.close()

with open(output_file_name,'w') as f_out:
    for w_list in output_corpus:
        if tokenize_sents == 2:
            f_out.write('\t'.join(w_list)+'\n')
        else:
            f_out.write(' '.join(w_list)+'\n')
#pipeline = ["sentencizer", "tokenizer"]
#lang = "en"
#data_path = "/home/hschang/anaconda3/lib/python3.7/site-packages/spacy/data/en/en_core_web_sm-2.1.0"

#raw_text = 'Hello, world. Here are two sentences.'
#nlp = English()
#nlp = spacy.load('en_core_web_sm')

#sentencizer = Sentencizer()

#sentencizer = nlp.create_pipe("sentencizer")
#nlp.add_pipe(sentencizer)

#cls = spacy.util.get_lang_class(lang) 
#nlp = cls()
#nlp.from_disk(data_path)
#print(doc)
#doc = nlp(raw_text, disable=['parser', 'tagger', 'ner'])
#sentences = [sent.string.strip() for sent in doc.sents]
#sentences = [sent.tokens for sent in doc.sents]

#print(sentences)
