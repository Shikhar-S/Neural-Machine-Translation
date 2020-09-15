import math
#Make directories needed for running in guild run folder
import os
print('Current run folder : ',os.getcwd())
os.makedirs(os.path.join(os.getcwd(),'tblogs/'),exist_ok=True)
os.makedirs(os.path.join(os.getcwd(),'trained_models/'),exist_ok=True)
#########################################################################
import utils
import torch
import time
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from datareader import DataReader, collator
from models.bert_transformer import BertNMTTransformer
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle
from transformers import BertTokenizer

#-------------------------------------------------------------------------------------------------
import argparse


writer = None
logger = utils.get_logger()

def str2bool(v):
    return v.lower() in ('true')

def str2dict(v):
    return {'run': str(v) }

def str2tuple(v):
    v = v.split('`!`!`')
    return (v[0],v[1])

def log_parsed_args(args):
    for arg, value in sorted(vars(args).items()):
        logger.info("Argument %s: %r", arg, value, extra=args.exec_id)

parser = argparse.ArgumentParser()
parser.add_argument("--batch",type=int,default=32)
parser.add_argument("--epochs",type=int,default=10)
parser.add_argument("--lr",type=float,default=1e-4)

parser.add_argument("--bert_model",default='bert-base-cased')
parser.add_argument('--padding',default='max_length')
parser.add_argument('--truncation',type=str2bool,default=True)

parser.add_argument("--d_model",type=int,default=512)
parser.add_argument("--n_encoder_layers",type=int,default=6)
parser.add_argument("--n_head",type=int,default=8)
parser.add_argument("--n_decoder_layers",type=int,default=6)
parser.add_argument("--dim_feedfwd",type=int,default=2048)
parser.add_argument("--dropout",type=float,default=0.1)
parser.add_argument('--bertinit',type=str2bool,default=False)

parser.add_argument("--device",type=str,default='auto',choices=['cpu', 'gpu','auto'])
parser.add_argument('--exec_id',type=str2dict,default={'run': '_guild_' + str(time.time()).replace('.','')})

parser.add_argument('--training_data',type=str2tuple,default=('./Data/processed_data/train.en','./Data/processed_data/train.cmd.template'))
parser.add_argument('--testing_data',type=str2tuple,default=('./Data/processed_data/test.en','./Data/processed_data/test.cmd.template'))
parser.add_argument('--validation_data',type=str2tuple,default=('./Data/processed_data/valid.en','./Data/processed_data/valid.cmd.template'))

parser.add_argument('--save_model_path',type=str,default='./trained_models/transformer')
parser.add_argument('--trg_vocab_path',type=str,default='./trained_models/transformer_dic.pickle')

parser.add_argument('--save_checkpoint',type=str2bool,default=False)
parser.add_argument('--load_checkpoint',type=str2bool,default=False)
parser.add_argument('--checkpoint_path',type=str,default='./trained_models/chkpt_transformer')

parser.add_argument('--mode',type=str,default='train',choices=['train','infer'])
parser.add_argument('--load_model_path',type=str,default='./trained_models/transformer.pt')
parser.add_argument('--max_len',type=int,default=30)
parser.add_argument('--output_file',type=str,default='./translation_out.txt')
parser.add_argument('--gen_test_translations',type=str2bool,default=False)

def get_args():
    args,unparsed = parser.parse_known_args()
    if args.save_checkpoint and args.mode != 'infer':
        args.checkpoint_path = args.checkpoint_path + args.exec_id['run'] + '.pt'
        print('Saving/Loading checkpoint at/from:',args.checkpoint_path)
    
    if args.mode == 'train':
        args.save_model_path = args.save_model_path+args.exec_id['run']+ '.pt'
        writer = SummaryWriter('tblogs/'+args.exec_id['run'])
        print('Saving model at: ',args.save_model_path)

    log_parsed_args(args)
    return args, unparsed

#-----------------------------------------------------------------------------------------------

training_batch_ctr = 0
valid_batch_ctr = 0

def train(model, iterator, epoch, optimizer, criterion, clip, args,checkpoint=None):
    global training_batch_ctr
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
        src, src_pad_mask = batch[0]
        trg = batch[1]
        src=src.to(device)
        src_pad_mask=src_pad_mask.to(device)
        trg = trg.to(device)
        
        optimizer.zero_grad()
        output = model(src)
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
        #Save checkpoint after every 100 batches
        if batch_ctr%100==0 and args.save_checkpoint:
            torch.save({
            'epoch': epoch,
            'batch': batch_ctr,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch_loss': epoch_loss,
            }, args.checkpoint_path)
            av_loss=epoch_loss/(batch_ctr+1)
            writer.add_scalar('Batch Training loss',av_loss,training_batch_ctr)
            writer.add_scalar('Batch Training PPL',math.exp(av_loss),training_batch_ctr)
        batch_ctr+=1
        training_batch_ctr+=1
        
    return epoch_loss / (batch_ctr)

def evaluate(model, iterator, criterion, args,log_tb=True):
    if log_tb:
        global valid_batch_ctr
    device=utils.get_device(args)
    model.eval()
    
    epoch_loss = 0
    batch_ctr=0
    with torch.no_grad():
    
        for batch in tqdm(iterator):

            src, src_pad_mask = batch[0]
            trg = batch[1]
            src=src.to(device)
            src_pad_mask=src_pad_mask.to(device)
            trg = trg.to(device)

            output = model(src) 

            #trg = [trg sent len, batch size]
            #output = [trg sent len, batch size, output dim]

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            #trg = [(trg sent len - 1) * batch size]
            #output = [(trg sent len - 1) * batch size, output dim]

            loss = criterion(output, trg)

            epoch_loss += loss.item()
            batch_ctr+=1
            if log_tb:
                av_loss = epoch_loss/(batch_ctr)
                writer.add_scalar('Batch Validation loss',av_loss,valid_batch_ctr)
                writer.add_scalar('Batch Validation PPL',math.exp(av_loss),valid_batch_ctr)
                valid_batch_ctr+=1
            

        
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
    PAD_IDX = vocab['[PAD]']
    SOS_IDX = vocab['[CLS]']
    EOS_IDX = vocab['[SEP]']
    device = utils.get_device(args)

    model = BertNMTTransformer(args,SRC_VOCAB_SIZE,TRG_VOCAB_SIZE, PAD_IDX, SOS_IDX, EOS_IDX).to(device)
    model.load_state_dict(torch.load(args.load_model_path,map_location=torch.device(args.device)))
    
    if args.gen_test_translations:
        print('Generating outputs...')
        with open(args.testing_data[0],'r') as test_file,open(args.output_file,'w') as out_file:
            for sentence in test_file:
                translation,attention,translation_tokens = translate_sentence(model,tokenizer,sentence,args)
                print(translation,file=out_file)
                
        print('Done!')
    else:
        while True:
            sentence=input('Enter natural language instruction\n')
            if sentence=='exit':
                break
            translation,attention,translation_tokens = translate_sentence(model,tokenizer,sentence,args)
            print(translation)
            print(translation_tokens)
            with open(args.output_file,'w',encoding='UTF-8') as F:
                print('Translated: ',translation,file=F)
    #display_attention(tokenizer.tokenize(sentence),translation_tokens,attention)    

def training_mode(args):
    #Get Data
    MAX_LEN = args.max_len
    src_tokenizer=BertTokenizer.from_pretrained(args.bert_model)

    
    training_dataset = DataReader(args,args.training_data,src_tokenizer)
    with open(args.trg_vocab_path,'wb') as f:
        pickle.dump(training_dataset.trg_tokenizer,f)
    
    validation_dataset = DataReader(args,args.validation_data,src_tokenizer,trg_tokenizer = training_dataset.trg_tokenizer)

    TRG_VOCAB_SIZE = training_dataset.trg_tokenizer.trg_vocab_len
    SRC_VOCAB_SIZE = len(src_tokenizer.vocab)
    PAD_IDX = src_tokenizer.vocab['[PAD]']

    device = utils.get_device(args)

    training_dataloader = DataLoader(training_dataset, batch_size = args.batch, drop_last=True, collate_fn=collator)
    validation_dataloader = DataLoader(validation_dataset,batch_size = args.batch, drop_last=True, collate_fn=collator)

    #Get model
    model = BertNMTTransformer(SRC_VOCAB_SIZE, TRG_VOCAB_SIZE, args.d_model,args.n_head,args.n_encoder_layers,args.n_decoder_layers,args.dim_feedfwd,args.dropout,bert_on=args.bertinit).to(device)
    logger.info(model.apply(utils.init_weights),extra=args.exec_id) #init model
    logger.info("Number of trainable parameters: "+str(utils.count_parameters(model)),extra=args.exec_id) #log Param count

    #Train and Evaluate model
    N_EPOCHS = args.epochs
    CLIP = 1
    best_valid_loss = float('inf')

    optimizer = optim.Adam(model.parameters(),lr=args.lr)
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
        
        writer.add_scalars('Epoch losses',{'Epoch training loss':train_loss,'Epoch Validation loss':valid_loss},epoch)
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
        print('-----------------------------------------')
        

if __name__ == '__main__':
    args,unparsed = get_args()
    if len(unparsed)>0:
        logger.warning('Unparsed args: %s',unparsed)
    
    if args.mode == 'infer':
        inference_mode(args)
    elif args.mode == 'train':
        training_mode(args)
        #close tensorboard writer    
        writer.close()
