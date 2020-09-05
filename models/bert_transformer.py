import torch
import torch.nn as nn
from bert_decoder import BertNMTDecoder,BertNMTDecoderLayer
from bert_encoder import BertNMTEncoder,BertNMTEncoderLayer

class BertNMTTransformer(nn.Transformer):
    def __init__(self, d_model = 512, nhead = 8, num_encoder_layers = 6,num_decoder_layers = 6, dim_feedforward = 2048, dropout = 0.1,activation = "relu", bert_on = False):
        self.bert_on = bert_on
        if self.bert_on:
            custom_encoder = BertNMTEncoderLayer(d_model, nhead, dim_feedforward,dropout,activation)
            custom_decoder = BertNMTDecoderLayer(d_model, nhead, dim_feedforward,dropout,activation)
            self.encoder = BertNMTEncoder(custom_encoder,num_encoder_layers)
            self.decoder = BertNMTDecoder(custom_decoder,num_decoder_layers)
            self._reset_parameters()
            self.d_model = d_model
            self.nhead = nhead
        else:
            super(BertTransformer, self).__init__(d_model,nhead,num_encoder_layers,num_decoder_layers,dim_feedforward,dropout,activation,custom_encoder,custom_decoder)

    def forward(self, src, tgt, bert_encoding, src_mask = None, tgt_mask = None, memory_mask = None, src_key_padding_mask = None, tgt_key_padding_mask = None, memory_key_padding_mask = None):
        
        if not self.bert_on:
            return super().forward(src, tgt, bert_encoding, src_mask , tgt_mask , memory_mask , src_key_padding_mask , tgt_key_padding_mask , memory_key_padding_mask )

        if src.size(1) != tgt.size(1):
            raise RuntimeError("the batch number of src and tgt must be equal")

        if src.size(2) != self.d_model or tgt.size(2) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        memory = self.encoder(src, bert_encoding, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt, memory, bert_encoding, tgt_mask=tgt_mask, memory_mask=memory_mask,tgt_key_padding_mask=tgt_key_padding_mask,memory_key_padding_mask=memory_key_padding_mask)
        return output

