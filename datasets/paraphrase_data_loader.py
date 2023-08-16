import os
import torch
from torchvision import transforms
from lib.config import cfg
from datasets.paraphrase_dataset import ParaphraseDataset
import samplers.distributed
import numpy as np

def sample_collate(batch):
    indices, input_seq, target_seq, source_seq = zip(*batch)
    
    indices = np.stack(indices, axis=0).reshape(-1)
    input_seq = torch.cat([torch.from_numpy(b) for b in input_seq], 0)
    target_seq = torch.cat([torch.from_numpy(b) for b in target_seq], 0)
    source_seq = torch.cat([torch.from_numpy(b[None,:]) for b in source_seq], 0)

    return indices, input_seq, target_seq, source_seq

def sample_collate_val(batch):
    indices, source_seq = zip(*batch)
    
    indices = np.stack(indices, axis=0).reshape(-1)
    source_seq = torch.cat([torch.from_numpy(b[None,:]) for b in source_seq], 0)

    return indices, source_seq


def load_train(distributed, epoch, paraphrase_set):
    sampler = samplers.distributed.DistributedSampler(paraphrase_set, epoch=epoch) \
        if distributed else None
    shuffle = cfg.DATA_LOADER.SHUFFLE if sampler is None else False
    
    loader = torch.utils.data.DataLoader(
        paraphrase_set, 
        batch_size = cfg.TRAIN.BATCH_SIZE,
        shuffle = shuffle, 
        num_workers = cfg.DATA_LOADER.NUM_WORKERS, 
        drop_last = cfg.DATA_LOADER.DROP_LAST, 
        pin_memory = cfg.DATA_LOADER.PIN_MEMORY,
        sampler = sampler, 
        collate_fn = sample_collate
    )
    return loader

def load_val(split, n_sample, sentIds):
        
    paraphrase_set = ParaphraseDataset(            
        split = split,
        n_sample = n_sample,
        sentIds = sentIds, 
        seq_per_sample = cfg.DATA_LOADER.SEQ_PER_IMG
    )

    loader = torch.utils.data.DataLoader(
        paraphrase_set, 
        batch_size = cfg.TEST.BATCH_SIZE,
        shuffle = False, 
        num_workers = cfg.DATA_LOADER.NUM_WORKERS, 
        drop_last = False, 
        pin_memory = cfg.DATA_LOADER.PIN_MEMORY, 
        collate_fn = sample_collate_val
    )
    return loader