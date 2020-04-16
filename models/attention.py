import torch.nn as nn
import torch.nn.functional as F
import torch

class Attention(nn.Module):
    def __init__(self,encoder_dim,decoder_dim):
        super(Attention,self).__init__()
        self.a = nn.Linear(encoder_dim*2 + decoder_dim,decoder_dim)
        self.v = nn.Parameter(torch.rand(decoder_dim))
        #attention computing model applies linear layer on concatenated encode, decoder inputs
        #and then multiplies with a parameter to get dimension down to 1.

    def forward(self,decoder_hidden,encoder_hiddens,mask):
        #calculate e_ij
        max_inp_sentence_length = encoder_hiddens.shape[0]
        batch_size = encoder_hiddens.shape[1]

        #step1 : linear layer
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1,max_inp_sentence_length,1)
        encoder_hiddens = encoder_hiddens.permute(1,0,2)
        energy = torch.tanh(self.a(torch.cat((decoder_hidden,encoder_hiddens),dim=2)))

        #step2 : mul with parameter to reduce dim
        energy = energy.permute(0,2,1)
        v = self.v.repeat(batch_size, 1).unsqueeze(1)

        attention = torch.bmm(v, energy).squeeze(1)

        #mask pad tokens
        attention = attention.masked_fill(mask == 0, -1e10)
        
        return F.softmax(attention, dim = 1)