import argparse
import logging
import utils
import time

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
parser.add_argument("--bert_model",default='bert-base-cased')
parser.add_argument('--padding',default='max_length')
parser.add_argument('--truncation',type=str2bool,default=True)
parser.add_argument("--input_embedding_dim",type=int,default=128)
parser.add_argument("--output_embedding_dim",type=int,default=128)
parser.add_argument("--encoder_dim",type=int,default=512)
parser.add_argument("--decoder_dim",type=int,default=512)
parser.add_argument("--dropout",type=float,default=0.5)
parser.add_argument("--epochs",type=int,default=10)
parser.add_argument("--device",type=str,default='auto',choices=['cpu', 'gpu','auto'])
parser.add_argument('--exec_id',type=str2dict,default={'run': str(time.time()).replace('.','')})

parser.add_argument('--training_data',type=str2tuple,default=('./Data/processed_data/raw.en','./Data/processed_data/raw.cmd'))
parser.add_argument('--testing_data',type=str2tuple,default=('./Data/processed_data/raw.en','./Data/processed_data/raw.cmd'))
parser.add_argument('--validation_data',type=str2tuple,default=('./Data/processed_data/raw.en','./Data/processed_data/raw.cmd'))

parser.add_argument('--save_model_path',type=str,default='./trained_models/seq2seq.pt')
parser.add_argument('--save_dic_path',type=str,default='./trained_models/dictionary.pkl')
parser.add_argument('--save_checkpoint',type=str2bool,default=True)
parser.add_argument('--load_checkpoint',type=str2bool,default=False)
parser.add_argument('--checkpoint_path',type=str,default='./trained_models/checkpoint.pt')

parser.add_argument('--mode',type=str,default='train',choices=['train','infer','test'])
parser.add_argument('--load_model_path',type=str,default='./trained_models/seq2seq.pt')
parser.add_argument('--load_dic_path',type=str,default='./trained_models/dictionary.pkl')
parser.add_argument('--max_len',type=int,default=30)
parser.add_argument('--output_file',type=str,default='./translation_out.txt')

def get_args():
    args,unparsed = parser.parse_known_args()
    logger.info('__INIT__',extra=args.exec_id)
    log_parsed_args(args)
    return args, unparsed