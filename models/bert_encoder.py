import torch
import torch.nn as nn

class BertNMTEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(BertNMTEncoderLayer, self).__init__(d_model,nhead,dim_feedforward,dropout,activation)
        self.bert_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

    def forward(self, src, bert_encoding, src_mask = None, src_key_padding_mask = None):
        bert_attended = self.bert_attn(src, bert_encoding,bert_encoding)[0]
        src_attended = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src2 = 0.5 * (bert_attended + src_attended)

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class BertNMTEncoder(nn.TransformerEncoder):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(BertNMTEncoder, self).__init__(encoder_layer, num_layers, norm)
    
    def forward(self, src, bert_encoding, mask = None, src_key_padding_mask = None):
        output = src

        for mod in self.layers:
            output = mod(output, bert_encoding, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


        
