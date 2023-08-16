import os
import random
import numpy as np
import torch
import torch.utils.data as data
import lib.utils as utils
import pickle

class ParaphraseDataset(data.Dataset):
    def __init__(
        self,
        split=None,
        n_sample=None,
        data_folder=None,
        seq_per_sample=None,
        sentIds=None
    ):
        self.seq_per_sample = seq_per_sample
        self.data_folder = data_folder

        self.split = split
        self.n_sample = n_sample
        
        #self.without_target = without_target
        self.without_target = (not (split == "train"))
        
        if (sentIds is None):
            assert(data_folder is not None)
            with open(data_folder, "rb") as f:
                self.sentIds = pickle.load(f)
        else:
            self.sentIds = sentIds
            
        self.seq_len = len(self.sentIds['train_0']['source_seq'])
        #self.input_seq = 
        #self.target_seq = 
        #self.source_seq = 
 
    def set_seq_per_sample(self, seq_per_sample):
        self.seq_per_sample = seq_per_sample

    def __len__(self):
        return self.n_sample

    def __getitem__(self, index):
        sample_id = self.split + '_' + str(index)
        indices = np.array([index]).astype('int')

        source_seq = self.sentIds[sample_id]['source_seq']

        if self.without_target:
            return indices, source_seq

        input_seq = np.zeros((self.seq_per_sample, self.seq_len+1), dtype='int')
        target_seq = np.zeros((self.seq_per_sample, self.seq_len+1), dtype='int')
        
        n = len(self.sentIds[sample_id]['input_seq'])
        if n >= self.seq_per_sample:
            sid = 0
            ixs = random.sample(range(n), self.seq_per_sample)                
        else:
            sid = n
            ixs = random.sample(range(n), self.seq_per_sample - n)
            input_seq[0:n, :] = self.sentIds[sample_id]['input_seq']
            target_seq[0:n, :] = self.sentIds[sample_id]['target_seq']
           
        for i, ix in enumerate(ixs):
            input_seq[sid + i] = self.sentIds[sample_id]['input_seq'][ix,:]
            target_seq[sid + i] = self.sentIds[sample_id]['target_seq'][ix,:]
        return indices, input_seq, target_seq, source_seq