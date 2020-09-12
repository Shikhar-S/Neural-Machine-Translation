import logging
import torch
import torch.nn as nn

def get_logger(name=__file__, level=logging.INFO,filename='log/transformer.log'):    
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
    if args.device not in ['gpu','cpu','auto']:
        return args.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if(args.device == 'gpu' and not torch.cuda.is_available()):
        logger.error('Backend device: %s not available',args.device)
    if args.device != 'auto':
        device = torch.device('cpu'  if args.device=='cpu' else 'cuda')
    args.device=device
    return device

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)

def memReport():
    import gc
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            print(type(obj),obj.size())

def cpuStats():
    import os
    import sys
    import psutil
    print(sys.version)
    print(psutil.cpu_percent())
    print(psutil.virtual_memory())
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30
    print(memoryUse)
