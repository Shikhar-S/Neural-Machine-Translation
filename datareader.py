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

def collator(batch,PAD_IDX,max_src_len,max_trg_len):
    dyn_max_src_len = dyn_max_trg_len = 0
    for x,y in batch:
        dyn_max_src_len = max(dyn_max_src_len,len(x))
        dyn_max_trg_len = max(dyn_max_trg_len,len(y))
    
    dyn_max_src_len = min(dyn_max_src_len,max_src_len)
    dyn_max_trg_len = min(dyn_max_trg_len,max_trg_len)
    
    X=[]
    X_len= []
    Y=[]
    for x,y in batch:
        X.append(x[:dyn_max_src_len]+[PAD_IDX for i in range(max(dyn_max_src_len-len(x),0))])
        X_len.append(min(len(x),dyn_max_src_len))
        Y.append(y[:dyn_max_trg_len]+[PAD_IDX for i in range(max(dyn_max_trg_len-len(y),0))])
    
    Y=torch.tensor(Y).permute(1,0).contiguous()
    X=torch.tensor(X).permute(1,0).contiguous()
    X_len =torch.tensor(X_len)
    return (X,X_len),Y

class Vocab:
    def __init__(self,src_dic=None,trg_dic=None):
        self.src_stoi = src_dic
        self.src_itos = defaultdict(self.ret_unk)

        if self.src_stoi is not None:
            for k,v in self.src_stoi.items():
                self.src_itos[v]=k

        self.trg_stoi = trg_dic
        self.trg_itos = defaultdict(self.ret_unk)
        
        if self.trg_stoi is not None:
            for k,v in self.trg_stoi.items():
                self.trg_itos[v]=k
    
    def ret_z(self):
        return 0
    def ret_unk(self):
        return '<UNK>'
    
    def build_dic(self,path,preprocessor):
        dic=defaultdict(self.ret_z)
        dic['<sos>']=1
        dic['<eos>']=2
        dic['<pad>']=3
        ctr =  4
        with open(path,'r',encoding='UTF-8') as F:
            for line in F:
                tokens=preprocessor(line)
                for token in tokens:
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
    def __init__(self,args,paths,preprocessors,DIC=None):
        self.src_path = paths[0]
        self.trg_path = paths[1]
        self.src_preprocessor = preprocessors[0]
        self.trg_preprocessor = preprocessors[1]
        
        self.vocab = Vocab()
        if DIC is None:
            src_dic = self.vocab.build_dic(self.src_path,self.src_preprocessor)
            logger.info('Built source dictionary',extra=args.exec_id)
            trg_dic = self.vocab.build_dic(self.trg_path,self.trg_preprocessor)
            logger.info('Built target dictionary',extra=args.exec_id)
            self.vocab.add_src_dic(src_dic)
            self.vocab.add_trg_dic(trg_dic)
        else:
            self.vocab=DIC
        
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
        src_itr = open(self.src_path,encoding='UTF-8')
        trg_itr = open(self.trg_path,encoding='UTF-8')
        
        #Map each element using the line_mapper
        mapped_src_itr = map(lambda text : self.line_mapper(text,True), src_itr)
        mapped_trg_itr = map(lambda text : self.line_mapper(text,False), trg_itr)
        
        #Zip both iterators
        zipped_itr = zip(mapped_src_itr, mapped_trg_itr)
        
        return zipped_itr

# #TEST
# import config
# args,unparsed = config.get_args()
# test_dataset = DataReader(args,('./Data/dev_test/dev.en','./Data/dev_test/dev.hi'),(en_preprocessor,hi_preprocessor))
# dataloader = DataLoader(test_dataset, batch_size = 4, drop_last=True,collate_fn= lambda b: collator(b,3,2,50))

# ctr=0
# for X, y in dataloader:
#     ctr+=1
#     print(X)
#     print('-----')
#     print(y)
#     print(('#################'))
#     print(X[0].shape)
#     print(y.shape)
#     if ctr==2:
#         break