import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoder import Encoder
from models.decoder import Decoder
import random

class Seq2Seq(nn.Module):
    def __init__(self,args,input_vocab_sz,output_vocab_sz,pad_idx, sos_idx, eos_idx):
        super(Seq2Seq,self).__init__()
        self.input_vocab_sz = input_vocab_sz
        self.output_vocab_sz = output_vocab_sz
        self.encoder = Encoder(input_vocab_sz,args.input_embedding_dim,args.encoder_dim,args.decoder_dim)
        self.decoder = Decoder(output_vocab_sz,args.output_embedding_dim,args.encoder_dim,args.decoder_dim)
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.device = args.device
        
    def create_mask(self, src):
        mask = (src != self.pad_idx).permute(1, 0)
        return mask
        
    def forward(self, src, src_len, trg, teacher_forcing_ratio = 0.5):
     
        #src = [src sent len, batch size]
        #src_len = [batch size]
        #trg = [trg sent len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time
        
        if trg is None:
            assert teacher_forcing_ratio == 0, "Must be zero during inference"
            inference = True
            trg = torch.zeros((100, src.shape[1])).long().fill_(self.sos_idx).to(self.device)
        else:
            inference = False
            
        batch_size = src.shape[1]
        max_len = trg.shape[0] if trg is not None else 100
        
        #tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, self.output_vocab_sz).to(self.device)
        
        #tensor to store attention
        attentions = torch.zeros(max_len, batch_size, src.shape[0]).to(self.device)
        
        #encoder_outputs is all hidden states of the input sequence, back and forwards
        #hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_hiddens, hidden_last = self.encoder(src, src_len)
                
        #first input to the decoder is the <sos> tokens
        output = trg[0,:]
        
        mask = self.create_mask(src)
                
        #mask = [batch size, src sent len]
                
        for t in range(1, max_len):
            output, hidden_last, attention = self.decoder(output, hidden_last, encoder_hiddens, mask)
            outputs[t] = output
            attentions[t] = attention
            
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            output = (trg[t] if (teacher_force and not inference) else top1)
            if inference and output.item() == self.eos_idx:
                return outputs[:t], attentions[:t]
            
        return outputs, attentions