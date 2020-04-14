import logging
import torch

def get_logger(name=__file__, level=logging.INFO,filename='log/seq_to_seq.log'):    
    logger = logging.getLogger(name)

    if getattr(logger, '_init_done__', None):
        logger.setLevel(level)
        return logger

    logger._init_done__ = True
    logger.propagate = False
    logger.setLevel(level)

    formatter = logging.Formatter("%(asctime)s||%(run)s||%(levelname)s||%(message)s")
    handler = logging.FileHandler(filename,mode='a')
    handler.setFormatter(formatter)
    handler.setLevel(0)

    del logger.handlers[:]
    logger.addHandler(handler)

    return logger

logger = get_logger()

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def get_device(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if(args.device == 'gpu' and not torch.cuda.is_available()):
        logger.error('Backend device: %s not available',args.device)
    if args.device != 'auto':
        device = torch.device('cpu'  if args.device=='cpu' else 'cuda')
    
    args.device=device
    return device
