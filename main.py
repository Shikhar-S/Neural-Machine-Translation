import utils
import config
import torch
import time
import torch.optim as optim
import torch.nn as nn
from datareader import DataReader, en_preprocessor, hi_preprocessor, collator
from models.seq2seq import Seq2Seq
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

logger = utils.get_logger()

def train(model, iterator, optimizer, criterion, clip):
    
    model.train()
    
    epoch_loss = 0
    batch_ctr=0
    for i, batch in enumerate(tqdm(iterator)):
        
        src, src_len = batch[0]
        trg = batch[1]
        
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
    return epoch_loss / batch_ctr

def evaluate(model, iterator, criterion):
    
    model.eval()
    
    epoch_loss = 0
    batch_ctr=0
    with torch.no_grad():
    
        for i, batch in enumerate(tqdm(iterator)):

            src, src_len = batch[0]
            trg = batch[1]

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
        
    return epoch_loss / batch_ctr

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

def train_model(args):
    #Get Data
    training_dataset = DataReader(args.validation_data,en_preprocessor,hi_preprocessor)
    validation_dataset = DataReader(args.validation_data,en_preprocessor,hi_preprocessor,training_dataset.vocab)
    # testing_dataset = DataReader(args.testing_data,en_preprocessor,hi_preprocessor,training_dataset.vocab)
    
    INPUT_DIM = len(training_dataset.vocab.src_stoi)
    OUTPUT_DIM = len(training_dataset.vocab.trg_stoi)

    device = utils.get_device(args)

    PAD_IDX = training_dataset.vocab.src_stoi['<pad>']
    SOS_IDX = training_dataset.vocab.src_stoi['<sos>']
    EOS_IDX = training_dataset.vocab.src_stoi['<eos>']

    training_dataloader = DataLoader(training_dataset, batch_size = args.batch, drop_last=True, collate_fn=lambda b: collator(b,PAD_IDX))
    validation_dataloader = DataLoader(validation_dataset,batch_size = args.batch, drop_last=True, collate_fn=lambda b: collator(b,PAD_IDX))
    # testing_dataloader = DataLoader(testing_dataset,batch_size = args.batch, drop_last=True, collate_fn=lambda b: collator(b,PAD_IDX))

    #Get model
    model = Seq2Seq(args,INPUT_DIM,OUTPUT_DIM, PAD_IDX, SOS_IDX, EOS_IDX).to(device)


    #Train and Evaluate model
    N_EPOCHS = 10
    CLIP = 1
    best_valid_loss = float('inf')

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index = PAD_IDX)
    
    for epoch in range(N_EPOCHS): 
        start_time = time.time()
        
        train_loss = train(model, training_dataloader, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, validation_dataloader, criterion)
        
        end_time = time.time()
        
        epoch_mins, epoch_secs = utils.epoch_time(start_time, end_time)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), args.save_model)
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

if __name__ == '__main__':
    args,unparsed = config.get_args()
    if len(unparsed)>0:
        logger.warning('Unparsed args: %s',unparsed)
    train_model(args)