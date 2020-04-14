from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader
from collections import defaultdict
import torch
from indicnlp.tokenize import indic_tokenize
import utils

logger = utils.get_logger()

def en_preprocessor(text):
    return [t.lower().replace('.','') for t in text.split()]

def hi_preprocessor(text):
    return [token for token in indic_tokenize.trivial_tokenize(text)]

def collator(batch,PAD_IDX):
    max_src_len = max_trg_len = 0
    for x,y in batch:
        max_src_len = max(max_src_len,len(x))
        max_trg_len = max(max_trg_len,len(y))
    X=[]
    X_len= []
    Y=[]
    for x,y in batch:
        X.append(x+[PAD_IDX for i in range(max_src_len-len(x))])
        X_len.append(len(x))
        Y.append(y+[PAD_IDX for i in range(max_trg_len-len(y))])
    
    Y=torch.tensor(Y).permute(1,0).contiguous()
    X=torch.tensor(X).permute(1,0).contiguous()
    X_len =torch.tensor(X_len)
    return (X,X_len),Y

class Vocab:
    def __init__(self,src_dic=None,trg_dic=None):
        self.src_stoi = src_dic
        self.src_itos = defaultdict(lambda : '<UNK>')
        
        if self.src_stoi is not None:
            for k,v in self.src_stoi.items():
                self.src_itos[v]=k

        self.trg_stoi = trg_dic
        self.trg_itos = defaultdict(lambda : '<UNK>')
        
        if self.trg_stoi is not None:
            for k,v in self.trg_stoi.items():
                self.trg_itos[v]=k

    def build_dic(self,path,preprocessor):
            dic=defaultdict(lambda : 0)
            dic['<sos>']=1
            dic['<eos>']=2
            dic['<pad>']=3
            ctr =  4
            with open(path,'r') as F:
                for line in F:
                    for token in preprocessor(line):
                        if token not in dic:
                            dic[token]=ctr
                            ctr+=1
            return dic
    
    def add_src_dic(self,dic):
        self.src_stoi = dic
        for k,v in self.src_stoi.items():
            self.src_itos[v]=k
    
    def add_trg_dic(self,dic):
        self.trg_stoi = dic
        for k,v in self.trg_stoi.items():
            self.trg_itos[v]=k

class DataReader(IterableDataset):
    def __init__(self,paths,src_preprocessor,trg_preprocessor,DIC=None):
        self.src_path = paths[0]
        self.trg_path = paths[1]
        
        self.vocab = Vocab()
        if DIC is None:
            src_dic = self.vocab.build_dic(self.src_path,src_preprocessor)
            logger.info('Built source dictionary')
            trg_dic = self.vocab.build_dic(self.trg_path,trg_preprocessor)
            logger.info('Built target dictionary')
            self.vocab.add_src_dic(src_dic)
            self.vocab.add_trg_dic(trg_dic)
        else:
            self.vocab=DIC
        
        self.src_preprocessor = src_preprocessor
        self.trg_preprocessor = trg_preprocessor

    def line_mapper(self, line, is_src):
        text = line
        tokens = []
        if is_src:
            tokens.append(self.vocab.src_stoi['<sos>'])
            tokens = tokens + [self.vocab.src_stoi[token] for token in self.src_preprocessor(text)]
            tokens.append(self.vocab.src_stoi['<eos>'])
        else:
            tokens.append(self.vocab.trg_stoi['<sos>'])
            tokens = tokens + [self.vocab.trg_stoi[token] for token in self.trg_preprocessor(text)]
            tokens.append(self.vocab.trg_stoi['<eos>'])
        return tokens

    def __iter__(self):
        #Create an iterator
        src_itr = open(self.src_path)
        trg_itr = open(self.trg_path)
        
        #Map each element using the line_mapper
        mapped_src_itr = map(lambda text : self.line_mapper(text,True), src_itr)
        mapped_trg_itr = map(lambda text : self.line_mapper(text,False), trg_itr)
        
        #Zip both iterators
        zipped_itr = zip(mapped_src_itr, mapped_trg_itr)
        
        return zipped_itr

#TEST

# test_dataset = DataReader('./Data/dev_test/dev.en','./Data/dev_test/dev.hi',en_preprocessor,hi_preprocessor)
# print('built vocab')
# dataloader = DataLoader(test_dataset, batch_size = 2, drop_last=True, collate_fn=lambda b: collator(b,3))

# for X, y in dataloader:
#     print(X)
#     print()
#     print(y)
#     break
