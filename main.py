import math
import utils
import config
import torch
import time
import torch.optim as optim
import torch.nn as nn
from datareader import DataReader, collator
from models.bert2bert import Bert2Bert
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle
from transformers import BertTokenizer

logger = utils.get_logger()

def train(model, iterator, epoch, optimizer, criterion, clip, args,checkpoint=None):
    device=utils.get_device(args)
    model.train()
    
    epoch_loss = 0
    batch_ctr=0
    #load checkpoint, if applicable
    if checkpoint is not None:
        batch_ctr=checkpoint['batch']
        epoch_loss=checkpoint['epoch_loss']

    for current_batch_ctr,batch in enumerate(tqdm(iterator)):
        if current_batch_ctr < batch_ctr:
            continue 

        torch.cuda.empty_cache()
        # print('Allocated',torch.cuda.memory_allocated())
        # print('Cached',torch.cuda.memory_cached())
        src, src_mask = batch[0]
        trg = batch[1]
        src=src.to(device)
        src_mask=src_mask.to(device)
        trg = trg.to(device)
        
        optimizer.zero_grad()
        
        output = model(src, trg, src_mask)
        
        logits = output.logits

        trg = trg.view(-1)
        #trg [batch_size,seq_len]
        logits = logits.view(-1, logits.shape[-1])
        
        loss = criterion(logits, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        epoch_loss += loss.item()
        #Save checkpoint after every 100 batches
        if batch_ctr%100==0 and args.save_checkpoint:
            torch.save({
            'epoch': epoch,
            'batch': batch_ctr,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch_loss': epoch_loss,
            }, args.checkpoint_path)
            EPOCH_INFO=f'Epoch: {epoch+1:02} | Batch: {batch_ctr+1:02}'
            av_loss=epoch_loss/(batch_ctr+1)
            LOSS_INFO=f'\tRunning av training Loss: {av_loss:.3f} | Train PPL: {math.exp(av_loss):7.3f}'
            logger.info(EPOCH_INFO,extra=args.exec_id)
            logger.info(LOSS_INFO,extra=args.exec_id)
        batch_ctr+=1
        break
    return epoch_loss / (batch_ctr)

def evaluate(model, iterator, criterion, args):
    device=utils.get_device(args)
    model.eval()
    
    epoch_loss = 0
    batch_ctr=0
    with torch.no_grad():
    
        for batch in tqdm(iterator):

            src, src_mask = batch[0]
            trg = batch[1]

            src=src.to(device)
            src_mask=src_mask.to(device)
            trg = trg.to(device)

            output = model(src, src_mask, trg)

            logits = output.logits

            trg = trg.view(-1)
            #trg [batch_size,seq_len]
            logits = logits.view(-1, logits.shape[-1])
            
            loss = criterion(logits, trg)

            epoch_loss += loss.item()
            batch_ctr+=1
            break
        
    return epoch_loss / (batch_ctr)

def translate_sentence(model,vocab,sentence,args):
    model.eval()
    device = utils.get_device(args)
    tokenized = en_preprocessor(sentence)
    tokenized = ['<sos>'] + tokenized + ['<eos>']
    numericalized = [vocab.src_stoi.get(t,0) for t in tokenized]
    sentence_length = torch.LongTensor([len(numericalized)]).to(device) 
    tensor = torch.LongTensor(numericalized).unsqueeze(1).to(device) 
    translation_tensor_logits, attention = model(tensor, sentence_length, None,teacher_forcing_ratio=0) 
    translation_tensor = torch.argmax(translation_tensor_logits.squeeze(1), 1)
    translation = [vocab.trg_itos.get(t.item(),'<UNK>') for t in translation_tensor]
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
    with open(args.load_dic_path, 'rb') as F:
        vocab = pickle.load(F)
    
    VOCAB_SIZE = len(vocab.keys())
    PAD_IDX = vocab.src_stoi['[PAD]']
    SOS_IDX = vocab.src_stoi['[CLS]']
    EOS_IDX = vocab.src_stoi['[SEP]']
    device = utils.get_device(args)

    model = Seq2Seq(args,VOCAB_SIZE, PAD_IDX, SOS_IDX, EOS_IDX).to(device)
    model.load_state_dict(torch.load(args.load_model_path,map_location=torch.device(args.device)))

    sentence=input('Enter natural language instruction')
    translation,attention = translate_sentence(model,vocab,sentence,args)
    with open(args.output_file,'w',encoding='UTF-8') as F:
        print('Translated: ',' '.join(translation),file=F)
    display_attention(sentence,translation,attention)    

def training_mode(args):
    #Get Data
    MAX_LEN = args.max_len
    tokenizer=BertTokenizer.from_pretrained(args.bert_model)
    VOCAB_SIZE=len(tokenizer.vocab.keys())

    training_dataset = DataReader(args,args.training_data,tokenizer)
    validation_dataset = DataReader(args,args.validation_data,tokenizer)
    

    device = utils.get_device(args)

    PAD_IDX = tokenizer.vocab['[PAD]']
    SOS_IDX = tokenizer.vocab['[CLS]']
    EOS_IDX = tokenizer.vocab['[SEP]']
    
    training_dataloader = DataLoader(training_dataset, batch_size = args.batch, drop_last=True, collate_fn=collator)
    validation_dataloader = DataLoader(validation_dataset,batch_size = args.batch, drop_last=True, collate_fn=collator)

    #Get model
    model = Bert2Bert(args).to(device)
    logger.info("Number of trainable parameters: "+str(utils.count_parameters(model)),extra=args.exec_id) #log Param count

    #Train and Evaluate model
    N_EPOCHS = args.epochs
    CLIP = 1
    best_valid_loss = float('inf')

    optimizer = optim.Adam(model.parameters())
    #optimizer = optim.SGD(model.parameters(),lr=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index = PAD_IDX)
    
    #load checkpoint, if applicable
    start_epoch = 0
    checkpoint=None
    if args.load_checkpoint:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']


    for epoch in range(start_epoch,N_EPOCHS): 
        start_time = time.time()
        
        train_loss = train(model, training_dataloader, epoch, optimizer, criterion, CLIP, args, checkpoint)
        valid_loss = evaluate(model, validation_dataloader, criterion, args)
        
        end_time = time.time()
        
        epoch_mins, epoch_secs = utils.epoch_time(start_time, end_time)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), args.save_model_path)
            with open(args.save_dic_path,'wb') as F:
                pickle.dump(tokenizer.vocab,F)
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
        print('-----------------------------------------')
        break

if __name__ == '__main__':
    args,unparsed = config.get_args()
    if len(unparsed)>0:
        logger.warning('Unparsed args: %s',unparsed)
    
    if args.mode == 'infer':
        inference_mode(args)
    elif args.mode == 'train':
        training_mode(args)
