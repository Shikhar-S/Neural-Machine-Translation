import spacy
import sys
from tqdm import tqdm

from indicnlp.tokenize import indic_tokenize

nlp = spacy.load('en', parser=False, entity=False)

path= sys.argv[1]
write_path= sys.argv[2]
is_src= int(sys.argv[3])==1

wf= open(write_path,'w')

def tokenize_src(text):
    return text.split()
    return [token.orth_ for token in nlp(text)]

def tokenize_trg(text):
    return [token for token in indic_tokenize.trivial_tokenize(text)]

with open(path,'r') as F:
    for j,line in enumerate(tqdm(F)):
        tokens=tokenize_src(line) if is_src else tokenize_trg(line)
        for i,t in enumerate(tokens):
            endc=' '
            if i==len(tokens)-1:
                endc=''
            print(t,file=wf,end=endc)

wf.close()