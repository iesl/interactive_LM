import spacy
from spacy.lang.en import English
import argparse

class ngram:

    def __init__(self):
        self.unigram_list = []
        self.bigram_list = []


    def ngram_list(self, n, doc):
        res = []
        for i in range(len(doc)-n+1):
            string = ""
            for j in range(n-1):
                string += doc[i+j].text+" "
            string += doc[i+n-1].text
            res.append(string)
        return res

    
    def add(self, string):
        nlp = English()
        doc = nlp(string)
        self.unigram_list.extend(self.ngram_list(1,doc))
        self.bigram_list.extend(self.ngram_list(2,doc))

    def diversity_n(self):
        if len(self.unigram_list) == 0:
            return 0, 0
        #print(self.bigram_list)
        return len(set(self.unigram_list))/(1e-15 + len(self.unigram_list)), len(set(self.bigram_list))/(1e-15+len(self.bigram_list))

def main():
    n = ngram()
    n.add("I am who I am, am I the who? Oh, I am!")
    n.add("I love it I love it I love it I love it")
    n.add("In October 2012, Steyer stepped down from his position at Farallon in order to focus on advocating for alternative energy.[18][19] Steyer decided to dispose of his carbon-polluting investments in 2012, although critics say he did not dispose of them quickly enough and noted that the lifespan of the facilities he funded would extend through 2030")
    print(n.diversity_n())


if __name__ == "__main__":
    main()
