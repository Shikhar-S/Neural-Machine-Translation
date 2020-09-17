import torch
import torch.nn.functional as F
import numpy as np


def beam_search(translation_tensor_logits,beams=5):
    ts_prob = F.softmax(translation_tensor_logits,dim=2)
    ts_prob = ts_prob.squeeze(1).detach().numpy()
    sequences = [(list(),0)]
    for current_step_prob in ts_prob:
        new_sequence_score = []
        for token_list,score in sequences:
            for token_id,token_prob in enumerate(current_step_prob):
                new_score = np.log(token_prob) + score
                new_token_list = token_list  +[token_id]
                new_sequence_score.append((new_token_list,new_score))
        sequences = new_sequence_score
        sequences = sorted(sequences,key = lambda x : -x[1])[:beams]
    return sequences