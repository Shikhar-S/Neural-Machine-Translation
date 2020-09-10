from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader
from collections import defaultdict
import torch
import utils
from transformers import BertTokenizer

logger = utils.get_logger()

def collator(batch):
    #print(batch)
    x=[]
    y=[]
    x_len=[]
    for item_x,item_y in batch:
        x.append(item_x['input_ids'])
        y.append(item_y['input_ids'])
        x_len.append(sum(item_x['attention_mask']))  #attention mask has 1 for unpadded token
    
    x_len = torch.tensor(x_len,dtype=torch.long).contiguous()
    x = torch.tensor(x,dtype=torch.long).permute(1,0).contiguous()
    y = torch.tensor(y,dtype=torch.long).permute(1,0).contiguous()

    return (x,x_len),y

class Vocab:
    # 0 is pad , 100-102 reserved
    # len returned on 
    def __init__(self,src_tokenizer,trg_path):
        self.path = trg_path
        self.trg_stoi = {}
        self.trg_itos = {}
        
        self.pad = src_tokenizer.vocab['[PAD]'] # 0
        self.unknown_id = src_tokenizer.vocab['[UNK]']
        self.sos = src_tokenizer.vocab['[CLS]'] #101
        self.eos = src_tokenizer.vocab['[SEP]'] #102

        self.trg_stoi['[PAD]'] = self.pad
        self.trg_stoi['[CLS]'] = self.sos
        self.trg_stoi['[SEP]'] = self.eos
        self.trg_stoi['[UNK]'] = self.unknown_id

        self.trg_vocab_len = 1
        self.build_dic()
        self.trg_vocab_len += 1 # 1 for 0 based indexing on the embedidng matrix.
        assert self.trg_vocab_len >= 103, 'will not work correctly.'
    
    def get_id(self,token):
        return self.trg_stoi.get(token,self.unknown_id)
    
    def get_token(self,id):
        return self.trg_itos.get(id,'[UNK]')
    
    def update_src_vocab(self,token):            
        self.trg_stoi[token] = self.trg_vocab_len
        self.trg_vocab_len += 1
        while self.trg_vocab_len==self.sos or self.trg_vocab_len==self.eos or self.trg_vocab_len==self.pad:
            self.trg_vocab_len += 1
    
    def build_dic(self):
        with open(self.path,'r',encoding='UTF-8') as F:
            for line in F:
                tokens=line.strip().split()
                for token in tokens:
                    self.update_src_vocab(token)
        self._build_inv_trg_dic()
    
    def _build_inv_trg_dic(self):
        for k,v in self.trg_stoi.items():
            self.trg_itos[v]=k


class DataReader(IterableDataset):
    def __init__(self,args,paths,src_tokenizer,trg_tokenizer=None):
        self.src_path = paths[0]
        self.trg_path = paths[1]
        self.src_tokenizer=src_tokenizer
        if trg_tokenizer is None:
            self.trg_tokenizer = Vocab(self.src_tokenizer,paths[1])
        else:
            self.trg_tokenizer = trg_tokenizer

        self.padding = args.padding
        self.max_length = args.max_len

    def src_line_mapper(self, text):
        tokens = self.src_tokenizer(text,padding='max_length',max_length=self.max_length,truncation=True)
        return tokens
        
    
    def trg_line_mapper(self,text):
        text_tokens = text.strip().split()
        text_tokens_id = []
        for token in text_tokens:
            token = token.strip()
            text_tokens_id.append(self.trg_tokenizer.get_id(token))

        text_tokens_id = [self.trg_tokenizer.sos] + text_tokens_id
        
        text_tokens_id = text_tokens_id[:self.max_length-1]
        text_tokens_id = text_tokens_id + [self.trg_tokenizer.eos] + [self.trg_tokenizer.pad]*(max(0,self.max_length-1 - len(text_tokens_id)))
        return {'input_ids': text_tokens_id }


    def __iter__(self):
        #Create an iterator
        src_itr = open(self.src_path,encoding='UTF-8')
        trg_itr = open(self.trg_path,encoding='UTF-8')
        
        #Map each element using the line_mapper
        mapped_src_itr = map(lambda text : self.src_line_mapper(text), src_itr)
        mapped_trg_itr = map(lambda text : self.trg_line_mapper(text), trg_itr)
        
        #Zip both iterators
        zipped_itr = zip(mapped_src_itr, mapped_trg_itr)
        
        return zipped_itr

# #TEST
# import config
# args,unparsed = config.get_args()
# test_dataset = DataReader(args,('./Data/processed_data/train.en','./Data/processed_data/train.cmd'),BertTokenizer.from_pretrained('bert-base-cased'))
# print(test_dataset.trg_tokenizer.trg_vocab_len)
# dataloader = DataLoader(test_dataset, batch_size = 4, drop_last=True,collate_fn= collator)

# ctr=0
# for X, Y in dataloader:
#     ctr+=1
#     print(X[1])
#     print(X[0].shape)
#     print(Y.shape)
#     print(Y)
#     break
#     if ctr==2:
#          break
