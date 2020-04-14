import argparse
import logging
import utils
import time

logger = utils.get_logger()

def str2bool(v):
    return v.lower() in ('true')

def str2dict(v):
    return {'run': str(v) }


parser = argparse.ArgumentParser()
parser.add_argument("--batch",type=int,default=16)
parser.add_argument("--input_embedding_dim",type=int,default=128)
parser.add_argument("--output_embedding_dim",type=int,default=128)
parser.add_argument("--encoder_dim",type=int,default=512)
parser.add_argument("--decoder_dim",type=int,default=512)
parser.add_argument("--epochs",type=int,default=10)
parser.add_argument("--device",type=str,default='auto',choices=['cpu', 'gpu','auto'])
parser.add_argument('--train_split',type=float,default=0.95)
parser.add_argument('--exec_id',type=str2dict,default={'run': str(time.time()).replace('.','')})
parser.add_argument('--training_data',type=tuple,default=('./Data/parallel/IITB.en-hi.en','./Data/parallel/IITB.en-hi.hi'))
parser.add_argument('--testing_data',type=tuple,default=('./Data/dev_test/test.en','./Data/dev_test/test.hi'))
parser.add_argument('--validation_data',type=tuple,default=('./Data/dev_test/dev.en','./Data/dev_test/dev.hi'))
parser.add_argument('--save_model',type=str,default='./trained_models/seq2seq.pt')

def get_args():
    args,unparsed = parser.parse_known_args()
    logger.info('Parsed arguments',extra=args.exec_id)
    return args, unparsed