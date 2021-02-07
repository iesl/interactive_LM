# Reading and Changing the Mind of Transformers for Topically-Controllable Language Generation

We will first introduce the how to run the demo. Then, introduce how to run the training and evaluation code.



## Requirements

Python 3.x

PyTorch 1.x

IPython notebook (for running the interactive demo)

You might need to install some other python packages if you find code needs them but you haven't installed it. 


If your python directory is not \~/anaconda3/bin/python, please change your PY_PATH in the XXX

install spacy (for tokenization)

install nltk (for evaluation and pplm)


## Running IPython Notebook Demo

Download the pretrained models and GloVe embedding. 


## Preprocessing Wikipedia for Training and Evaluation

https://github.com/attardi/wikiextractor

bin/preprocessing_single_proc.sh

Align tokens from Spacy and from GPT2

In our paper, we use learning rate = 1e-4. You can also try other values between 1e-4 and 1e-5.


## Running Training

Download the word embedding (e.g., GloVe)

Option Generator
Conditional Text Generator


## Running Evaluation using Automatic Metrics



https://radimrehurek.com/gensim/wiki.html
LDA
src/preprocessing/tools/train_LDA_model.py

how to compute global word clusters
src/preprocessing/tools/word_emb_global_clustering.py

If you do not want to test them, just remove LDA_org and global_centers from XXX

Option Generator Visualization


warning the diversity only works as paper said when we set bsz = 1

## Running Evaluation using Amazon Mechanical Turk

Download STS

Generate CSV

Our crowdsourcing template could be found in 

## Citation
If you use the code in a publication, please cite our paper.
```
Haw-Shiuan Chang, Jiaming Yuan, Mohit Iyyer and Andrew McCallum,
“Reading and Changing the Mind of Transformers for Topically-Controllable Language Generation.” 
Conference of the European Chapter of the Association for Computational Linguistics (EACL), 2021
```