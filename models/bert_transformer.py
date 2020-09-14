import torch
import torch.nn as nn
from models.bert_decoder import BertNMTDecoder,BertNMTDecoderLayer
from models.bert_encoder import BertNMTEncoder,BertNMTEncoderLayer
from models.positional_encoding import PositionalEncoding
import math

class BertNMTTransformer(nn.Transformer):
    def __init__(self, input_vocab_sz, output_vocab_sz ,d_model = 512, nhead = 8, num_encoder_layers = 6,num_decoder_layers = 6, dim_feedforward = 2048, dropout = 0.1,activation = "relu", bert_on = False):
        self.bert_on = bert_on

        if self.bert_on:
            custom_encoder = BertNMTEncoderLayer(d_model, nhead, dim_feedforward,dropout,activation)
            custom_decoder = BertNMTDecoderLayer(d_model, nhead, dim_feedforward,dropout,activation)
            self.encoder = BertNMTEncoder(custom_encoder,num_encoder_layers)
            self.decoder = BertNMTDecoder(custom_decoder,num_decoder_layers)
            self._reset_parameters()
            self.nhead = nhead
        else:
            super(BertNMTTransformer, self).__init__(d_model,nhead,dim_feedforward=dim_feedforward,dropout=dropout)

        self.decoder_trg_mask = None
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.input_encoder = nn.Embedding(input_vocab_sz, d_model)
        self.d_model = d_model
        self.output_decoder = nn.Linear(d_model, output_vocab_sz)

        

    def forward(self, src, bert_encoding=None):
        
        #create source mask on first forward mask/size mismatch with current mask
        if self.decoder_trg_mask is None or self.decoder_trg_mask.size(0) != len(src):
            device = src.device
            mask = self.generate_square_subsequent_mask(len(src)).to(device)
            self.decoder_trg_mask = mask

        #use positional embedding and embedding layer on source sentence.
        src = self.input_encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)

        if not self.bert_on:
            output = super().forward(src, src, tgt_mask = self.decoder_trg_mask)
            output = self.output_decoder(output)
            return output

        else:
            #bert initialised
            memory = self.encoder(src, bert_encoding)
            output = self.decoder(src, memory, bert_encoding, tgt_mask=self.decoder_trg_mask)
            output = self.output_decoder(output)

            return output

