import math
import utils
import config
import torch
import time
import torch.optim as optim
import torch.nn as nn
from datareader import DataReader, en_preprocessor, hi_preprocessor, collator, Vocab
from models.seq2seq import Seq2Seq
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle

logger = utils.get_logger()

def train(model, iterator, optimizer, criterion, clip, args):
    device=utils.get_device(args)
    model.train()
    
    epoch_loss = 0
    batch_ctr=0
    for batch in tqdm(iterator):
        torch.cuda.empty_cache()
        # print('Allocated',torch.cuda.memory_allocated())
        # print('Cached',torch.cuda.memory_cached())
        src, src_len = batch[0]
        trg = batch[1]
        src=src.to(device)
        src_len=src_len.to(device)
        trg = trg.to(device)
        
        optimizer.zero_grad()
        
        output, _ = model(src, src_len, trg)
        #trg = [trg sent len, batch size]
        #output = [trg sent len, batch size, output dim]
        
        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)
        #trg = [(trg sent len - 1) * batch size]
        #output = [(trg sent len - 1) * batch size, output dim]
        
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        epoch_loss += loss.item()
        batch_ctr+=1
    return epoch_loss / (batch_ctr*args.batch)

def evaluate(model, iterator, criterion, args):
    device=utils.get_device(args)
    model.eval()
    
    epoch_loss = 0
    batch_ctr=0
    with torch.no_grad():
    
        for batch in tqdm(iterator):

            src, src_len = batch[0]
            trg = batch[1]

            src=src.to(device)
            src_len=src_len.to(device)
            trg = trg.to(device)

            output, _ = model(src, src_len, trg, 0) #turn off teacher forcing

            #trg = [trg sent len, batch size]
            #output = [trg sent len, batch size, output dim]

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            #trg = [(trg sent len - 1) * batch size]
            #output = [(trg sent len - 1) * batch size, output dim]

            loss = criterion(output, trg)

            epoch_loss += loss.item()
            batch_ctr+=1
        
    return epoch_loss / (batch_ctr*args.batch)

def translate_sentence(model,vocab,sentence,args):
    model.eval()
    device = utils.get_device(args)
    tokenized = en_preprocessor(sentence) 
    tokenized = ['<sos>'] + tokenized + ['<eos>']
    numericalized = [vocab.src_stoi[t] for t in tokenized] 
    sentence_length = torch.LongTensor([len(numericalized)]).to(device) 
    tensor = torch.LongTensor(numericalized).unsqueeze(1).to(device) 
    translation_tensor_logits, attention = model(tensor, sentence_length, None) 
    translation_tensor = torch.argmax(translation_tensor_logits.squeeze(1), 1)
    translation = [vocab.trg_itos[t] for t in translation_tensor]
    translation, attention = translation[1:], attention[1:]
    return translation, attention

def display_attention(candidate, translation, attention):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    
    attention = attention.squeeze(1).cpu().detach().numpy()
    
    cax = ax.matshow(attention, cmap='bone')
   
    ax.tick_params(labelsize=15)
    ax.set_xticklabels([''] + ['<sos>'] + [t.lower() for t in en_preprocessor(candidate)] + ['<eos>'], 
                       rotation=45)
    ax.set_yticklabels([''] + translation)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()

def inference_mode(args):
    vocab=None
    with open(args.load_dic_path, 'rb') as F:
        vocab = pickle.load(F)

    INPUT_DIM = len(vocab.src_stoi)
    OUTPUT_DIM = len(vocab.trg_stoi)
    PAD_IDX = vocab.src_stoi['<pad>']
    SOS_IDX = vocab.src_stoi['<sos>']
    EOS_IDX = vocab.src_stoi['<eos>']
    device = utils.get_device(args)

    model = Seq2Seq(args,INPUT_DIM,OUTPUT_DIM, PAD_IDX, SOS_IDX, EOS_IDX).to(device)
    model.load_state_dict(torch.load(args.load_model_path))

    sentence=input('Enter sentence in source language')
    translation,attention = translate_sentence(model,vocab,sentence,args)
    print('Translated: ',' '.join(translation.join))
    display_attention(sentence,translation,attention)    

def training_mode(args):
    #Get Data
    TRG_MAX_LEN = args.trg_max_len
    SRC_MAX_LEN = args.src_max_len
    preprocessors=(en_preprocessor,hi_preprocessor)
    lengths=(SRC_MAX_LEN,TRG_MAX_LEN)
    vocab_sz=(args.input_vocab,args.output_vocab)

    training_dataset = DataReader(args,args.training_data,preprocessors,vocab_sz)
    validation_dataset = DataReader(args,args.validation_data,preprocessors,DIC=training_dataset.vocab)
    # testing_dataset = DataReader(args,args.testing_data,en_preprocessor,hi_preprocessor,training_dataset.vocab)
    
    INPUT_DIM = len(training_dataset.vocab.src_stoi)
    OUTPUT_DIM = len(training_dataset.vocab.trg_stoi)

    device = utils.get_device(args)

    PAD_IDX = training_dataset.vocab.src_stoi['<pad>']
    SOS_IDX = training_dataset.vocab.src_stoi['<sos>']
    EOS_IDX = training_dataset.vocab.src_stoi['<eos>']
    
    training_dataloader = DataLoader(training_dataset, batch_size = args.batch, drop_last=True, collate_fn=lambda b: collator(b,PAD_IDX,SRC_MAX_LEN,TRG_MAX_LEN))
    validation_dataloader = DataLoader(validation_dataset,batch_size = args.batch, drop_last=True, collate_fn=lambda b: collator(b,PAD_IDX,SRC_MAX_LEN,TRG_MAX_LEN))
    # testing_dataloader = DataLoader(testing_dataset,batch_size = args.batch, drop_last=True, collate_fn=lambda b: collator(b,PAD_IDX))

    #Get model
    model = Seq2Seq(args,INPUT_DIM,OUTPUT_DIM, PAD_IDX, SOS_IDX, EOS_IDX).to(device)
    logger.info(model.apply(utils.init_weights),extra=args.exec_id) #init model
    logger.info("Number of trainable parameters: "+str(utils.count_parameters(model)),extra=args.exec_id) #log Param count

    #Train and Evaluate model
    N_EPOCHS = args.epochs
    CLIP = 1
    best_valid_loss = float('inf')

    optimizer = optim.Adam(model.parameters())
    #optimizer = optim.SGD(model.parameters(),lr=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index = PAD_IDX)
    
    for epoch in range(N_EPOCHS): 
        start_time = time.time()
        
        train_loss = train(model, training_dataloader, optimizer, criterion, CLIP, args)
        valid_loss = evaluate(model, validation_dataloader, criterion, args)
        
        end_time = time.time()
        
        epoch_mins, epoch_secs = utils.epoch_time(start_time, end_time)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), args.save_model_path)
            with open(args.save_dic_path,'wb') as F:
                pickle.dump(training_dataset.vocab.src_stoi,F)
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

if __name__ == '__main__':
    args,unparsed = config.get_args()
    if len(unparsed)>0:
        logger.warning('Unparsed args: %s',unparsed)

    if args.mode == 'infer':
        inference_mode(args)
    elif args.mode == 'train':
        training_mode(args)
