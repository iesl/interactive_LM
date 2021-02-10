# Changing the Mind of Transformers for Topically-Controllable Language Generation

We will first introduce the how to run the IPython notebook demo by downloading our pretrained models. 
Then, we will introduce how to run our training and evaluation code.

![Image of our model](http://people.umass.edu/hawshiuancha/model_illustration_fig.png)

## Requirements

- An Unix like OS with at least one GPU
- To setup python environment, run `pip install -r requirements.txt`. I use python 3.7 and pytorch 1.3.1, but I think other python 3 or pytorch > 1.0 versions might also be fine or just require very simple revision of the code. Our codes also use IPython notebook (for running the interactive demo), Spacy (for tokenization), nltk (for running evaluation and pplm), and gensim (for running the LDA baseline).
- If your python directory is not `~/anaconda3/bin/python`, please change your PY_PATH in the all the scripts in `./bin`

## Running IPython Notebook Demo

- Download the pretrained models and dictionary file from [here](https://drive.google.com/drive/folders/1YKM-CbMPy7lBsqdwyvhjwBYMxaYkENbj?usp=sharing) or following the instructions for training code below
- Use IPython notebook to open `./src/evaluation/test_conditional_LM.ipynb`
- Run the 1st block after putting the models into the corresponding directory or revising the paths of TOPIC_MODEL_DIR, GENERATION_MODEL_DIR, DICT_FILE in the first block.
- Modify the input context prompt in the 2nd block and run the block to see the generated topics
- Choose some topics or specify some words and run the 3rd block to see the generated continuations that start with `conditional x:`. We will also generate the continuation without the condition that start with `original x:` as a baseline. The topical words that appear in the continuation will be highlighted.
- You can append a genearted continuation to the 2nd block and repeat the process 

## Preprocessing Wikipedia for Training and Evaluation

- First, download only the text from Wikipedia into json format using [WikiExtractor](https://github.com/attardi/wikiextractor)
- Check the path in `./bin/preprocessing_single_proc.sh` and run the script. In the preprocessing, we will run Spacy tokenizer and GPT2 tokenizer, heuristically align their resulting tokens, split the corpus into training/validation/testing sets, and store the word indices into tensors.
- Note that `./bin/preprocessing_single_proc.sh` might be slow because it does not parallelize the tokenization processes. If you use job scheduler like slurm in your server, you might want to see the parallized scripts for tokenization in `./bin/old/tokenize_all_wiki_gpt2.sh` and `./bin/old/tokenize_all_wiki.sh`

## Running Training

- Prepare a word embedding file (e.g., we download the GloVe embedding from [here](https://nlp.stanford.edu/projects/glove/))
- Train our option generator using `./bin/train_option_generator.sh`
- Train our conditional text generator using `./bin/train_conditional_generator.sh` (could train option generator and text generator at the same time)
- You can start from original GPT2 model or start from our pretrained models. In our paper, we use learning rate = 1e-4. You can also try other values between 1e-4 and 1e-5. 

## Running Evaluation using Automatic Metrics

- To evaluate/visualize conditional text generator, update the GENERATION_MODEL_DIR and TOPIC_MODEL_DIR using the model path from the previous step to run `./bin/train_conditional_generator.sh`. 
- To evaluate/visualize option generator, update the GENERATION_MODEL_DIR and TOPIC_MODEL_DIR and run `./bin/eval_option_generator.sh`. Set VISUALIZATION='Y' to visualize the topics given some randomly selected prompt. Set AUTO_EVAL_TOPICS='Y' to compare the quality of topics from different methods as we did in Table 1 in our EACL paper. Set AUTO_EVAL_GENRATION='Y' to evaluate the topics by the quality of text that is generated given these topics as we did in Table 6 in our paper appendix. 
- Our scores are stored at the end of each OUT_FILE file when AUTO_EVAL\*='Y'. Our text generator is called "model condition", and our option generator is called NSD_topic in our code, where NSD stands for neural set decoder.
- In our code, we also evaluate some globally clustering baselines such as LDA and kmeans. In order to test them, you can train a LDA model by following the steps [here](https://radimrehurek.com/gensim/wiki.html). You can also see an example code at `./src/preprocessing/tools/train_LDA_model.py`. For kmeans clustering, we use `./src/preprocessing/tools/word_emb_global_clustering.py`. If you do not want to test them, just remove LDA_org and global_centers from METHOD_LIST

## Running Evaluation using Amazon Mechanical Turk

- Download STSb dataset from [here](https://ixa2.si.ehu.eus/stswiki/index.php/STSbenchmark)
- Preprocessing STS using `./src/evaluation/filter_STS_for_GPT2.py`
- Set OUTPUT_CSV_FOR_MTURK='Y' in `./bin/train_conditional_generator.sh` and `./bin/eval_option_generator.sh` to generate CSV files for MTurk tasks.
- Our crowdsourcing templates and responses from workers could be found in `./MTurk_eval`

## Citation
If you use the code in a publication, please cite our paper.
```
Haw-Shiuan Chang, Jiaming Yuan, Mohit Iyyer, and Andrew McCallum,
“Changing the Mind of Transformers for Topically-Controllable Language Generation.” 
Conference of the European Chapter of the Association for Computational Linguistics (EACL), 2021
```
