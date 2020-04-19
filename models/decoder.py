import torch.nn as nn
import torch.nn.functional as F
import torch
from models.attention import Attention

class Decoder(nn.Module):
    def __init__(self,output_vocab_sz,output_embedding_dim,encoder_dim,decoder_dim,dropout):
        super(Decoder,self).__init__()
        self.embedding_layer = nn.Embedding(output_vocab_sz,output_embedding_dim)
        self.f = nn.GRU(output_embedding_dim + encoder_dim * 2  , decoder_dim)
        self.g = nn.Linear(output_embedding_dim + decoder_dim + encoder_dim * 2,output_vocab_sz)
        self.attention = Attention(encoder_dim,decoder_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self,input,decoder_hidden,encoder_hiddens,mask):
        #input = y_im1 , decoder_hidden = s_im1 , encoder_hiddens = h 
        
        #compute attention
        attn = self.attention(decoder_hidden,encoder_hiddens,mask)
        
        #compute weighted context vector c_i
        attn = attn.unsqueeze(1)
        encoder_hiddens = encoder_hiddens.permute(1, 0, 2)

        c_i = torch.bmm(attn, encoder_hiddens)
        
        #compute new decoder hidden state
        y_im1 = self.dropout(self.embedding_layer(input.unsqueeze(0)))
        c_i = c_i.permute(1,0,2)
        rnn_input = torch.cat((y_im1,c_i), dim=2 ) 

        s_i, s_i_copy = self.f(rnn_input,decoder_hidden.unsqueeze(0)) 

        assert (s_i==s_i_copy).all()
        
        #compute next token
        y_im1 = y_im1.squeeze(0)
        s_i = s_i.squeeze(0)
        c_i = c_i.squeeze(0)

        y_i = self.g(torch.cat((s_i, c_i, y_im1), dim = 1))

        return y_i, s_i_copy.squeeze(0), attn.squeeze(1)
