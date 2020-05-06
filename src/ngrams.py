from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
import argparse

parser = argparse.ArgumentParser(description='PyTorch Interactive LM')
parser.add_argument('--data', type=str, default='I am jiaming yuan, I am 20 year old, thanks I am')
args = parser.parse_args()

def ngrams(n, doc):
    res = []
    for i in range(len(doc)-n+1):
        string = ""
        for j in range(n-1):
            string += doc[i+j].text+" "
        string += doc[i+n-1].text
        res.append(string)
    return res


def diversity_n(string,n):
    nlp = English()
    doc = nlp(string)
    res = ngrams(n,doc)
    # for each in res:
    #     print(each)
    return len(set(res))/len(res)

print(diversity_n(args.data,3))
