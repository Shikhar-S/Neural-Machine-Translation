import torch
import torch.nn as nn
from bert_decoder import BertNMTDecoder,BertNMTDecoderLayer
from bert_encoder import BertNMTEncoder,BertNMTEncoderLayer

class BertNMTTransformer(nn.Transformer):
    def __init__(self, d_model = 512, nhead = 8, num_encoder_layers = 6,num_decoder_layers = 6, dim_feedforward = 2048, dropout = 0.1,activation = "relu", bert_on = False):
        self.bert_on = bert_on
        custom_encoder = BertNMTEncoderLayer(,bert_on)
        custome_decoder = BertNMTDecoderLayer(,bert_on)
        super(BertTransformer, self).__init__(d_model,nhead,num_encoder_layers,num_decoder_layers,dim_feedforward,dropout,activation,custom_encoder,custom_decoder)
    
    def __forward__(self,src,tgt,src_mask=None,tgt_mask=None,memory_mask=None,src_key_padding=None,tgt_key_padding=None,memory_key_padding=None):
        pass
