import logging
import gensim

#num_topics = 300
num_topics = 150
#wiki_dict_path = '/iesl/data/word_embedding/wikipedia/wiki2016_wordids.txt.bz2'
#wiki_tfidf_path = '/iesl/data/word_embedding/wikipedia/wiki2016_tfidf.mm'
#output_path = '/iesl/canvas/hschang/language_modeling/interactive_LM/models/LDA/LDA_wiki2016_'+str(num_topics)
wiki_dict_path = './data/raw/wikipedia/wiki2016_wordids.txt.bz2'
wiki_tfidf_path = './data/raw/wikipedia/wiki2016_tfidf.mm'
output_path = './models/LDA/LDA_wiki2016_'+str(num_topics)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# load id->word mapping (the dictionary), one of the results of step 2 above
id2word = gensim.corpora.Dictionary.load_from_text(wiki_dict_path)
# load corpus iterator
mm = gensim.corpora.MmCorpus(wiki_tfidf_path)

print(mm)
#MmCorpus(3931787 documents, 100000 features, 756379027 non-zero entries)
#lda = gensim.models.ldamodel.LdaModel(corpus=mm, id2word=id2word, num_topics=num_topics, update_every=1, passes=1)
lda = gensim.models.ldamulticore.LdaMulticore(corpus=mm, id2word=id2word, num_topics=num_topics, passes=1)
lda.print_topics(20)

lda.save(output_path)
