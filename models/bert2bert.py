from transformers import EncoderDecoderModel

class Bert2Bert:
    def __init__(self,args):
        self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(args.bert_model,args.bert_model)
    
    def __call__(self,input_ids,decoder_input_ids,attention_mask):
        return self.model(input_ids=input_ids,decoder_input_ids=decoder_input_ids,attention_mask=attention_mask,return_dict=True)
    
    def to(self,device):
        self.model = self.model.to(device)
        return self
    
    def parameters(self):
        return self.model.parameters()
    
    def train(self):
        self.model.train()
    
    def eval(self):
        self.model.eval()
    
    def state_dict(self):
        return self.model.state_dict()
    
    def load_state_dict(self,sd):
        self.model.load_state_dict(sd)