import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self,input_vocab_sz,input_embedding_dim,encoder_dim,decoder_dim):
        super(Encoder,self).__init__()
        self.embedding_layer=nn.Embedding(input_vocab_sz,input_embedding_dim)
        self.rnn = nn.GRU(input_embedding_dim,encoder_dim,bidirectional=True)
        self.forward_net = nn.Linear(encoder_dim * 2, decoder_dim)
    
    def forward(self,input,input_len):
        #embed input
        embeddings = self.embedding_layer(input)
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(embeddings,input_len,enforce_sorted=False)

        #feed into rnn to get all hidden states
        packed_hidden_states , last_hidden_state = self.rnn(packed_embeddings)
        hidden_states, _ = nn.utils.rnn.pad_packed_sequence(packed_hidden_states) #unpack

        #compute first hidden state for decoder
        last_hidden_state = torch.tanh(self.forward_net(torch.cat((last_hidden_state[-2,:,:], last_hidden_state[-1,:,:]), dim = 1)))

        return hidden_states, last_hidden_state

# import torch
# encoder = Encoder(12,18,24)
# inp = torch.tensor([[2,3,4,5,7],[4,5,6,1,1],[6,5,1,1,1]]).long()
# inpsz = torch.tensor([5,3,2])
# hid = encoder(inp.permute(1,0),inpsz)
# print(hid.shape)
# print(hid)


