import argparse
import logging
import utils
import time
#Todo: multiple execution model and dic save, automate with timestamp

logger = utils.get_logger()

def str2bool(v):
    return v.lower() in ('true')

def str2dict(v):
    return {'run': str(v) }

def log_parsed_args(args):
    for arg, value in sorted(vars(args).items()):
        logger.info("Argument %s: %r", arg, value, extra=args.exec_id)

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

parser.add_argument('--save_model_path',type=str,default='./trained_models/seq2seq.pt')
parser.add_argument('--save_dic_path',type=str,default='./trained_models/dictionary.pkl')

parser.add_argument('--train',type=str2bool,default='false')
parser.add_argument('--translate',type=str2bool,default='false')
parser.add_argument('--load_model_path',type=str,default='./trained_models/seq2seq.pt')
parser.add_argument('--load_dic_path',type=str,default='./trained_models/dictionary.pkl')

def get_args():
    args,unparsed = parser.parse_known_args()
    logger.info('__INIT__',extra=args.exec_id)
    log_parsed_args(args)
    return args, unparsed