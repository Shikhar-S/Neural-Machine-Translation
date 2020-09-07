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


class DataReader(IterableDataset):
    def __init__(self,args,paths,tokenizer):
        self.src_path = paths[0]
        self.trg_path = paths[1]
        self.tokenizer = tokenizer
        self.space = len(self.tokenizer.vocab)   
        self.padding = args.padding
        self.max_length = args.max_length

    def src_line_mapper(self, text):
        tokens = self.tokenizer(text,padding='max_length',max_length=self.max_length,truncation=True)
        return tokens
    
    def trg_line_mapper(self,text):
        tokens = text.strip().split()
        cmd = []
        for token in tokens:
            cmd.append(self.tokenizer.convert_tokens_to_ids(token.strip()))
            cmd.append(self.space)
        cmd = [self.tokenizer.vocab['[CLS]']] + cmd 
        cmd = cmd[:self.max_length-1]
        cmd = cmd + [self.tokenizer.vocab['[SEP]']] + [self.tokenizer.vocab['[PAD]']] * max(0, self.max_length-1 - len(cmd)) 
        return {'input_ids':cmd}

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
# dataloader = DataLoader(test_dataset, batch_size = 4, drop_last=True,collate_fn= collator)

# ctr=0
# for X, Y in dataloader:
#     ctr+=1
#     print(X[1])
#     print(X[0])
#     print(Y)
#     #print(y.keys())
#     break
#     if ctr==2:
#          break
