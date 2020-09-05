import torch
import torch.nn as nn

class BertNMTDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(BertNMTDecoderLayer, self).__init__(d_model, nhead, dim_feedforward,dropout,activation)
        self.bert_attn = MultiheadAttention(d_model,nhead,dropout = dropout)
    
    def forward(self, tgt, memory, bert_encoding, tgt_mask = None, memory_mask = None,tgt_key_padding_mask = None, memory_key_padding_mask = None):
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,key_padding_mask=memory_key_padding_mask)[0]
        tgt3 = self.bert_attn(tgt,bert_encoding,bert_encoding)[0]

        tgt_comb = 0.5*(tgt2+tgt3)

        tgt = tgt + self.dropout2(tgt_comb)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class BertNMTDecoder(nn.TransformerDecoder):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super(BertNMTDecoder, self).__init__(decoder_layer,num_layers,norm)
    
    def forward(self, tgt, memory, bert_encoding, tgt_mask = None,memory_mask = None, tgt_key_padding_mask = None,memory_key_padding_mask = None):
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, bert_encoding , tgt_mask=tgt_mask,memory_mask=memory_mask,tgt_key_padding_mask=tgt_key_padding_mask,memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output
