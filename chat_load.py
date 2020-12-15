from torch.utils.data.dataset import Dataset
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import torch
import numpy as np 

class post_set(Dataset):
    def __init__(self, post, tokenizer):
        #eos = [tokenizer.encoder["<|endoftext|>"]]
        with open(post) as f:
            table = f.readlines()
        temp = []
        m = []
        self.ll = []
        for l in table:
            srcs, tgt = l.strip().split('\t')
            temp_token = tokenizer.encode(srcs)
            temp_mask = [1 for i in range(len(temp_token))]
            if len(temp_token) >= 20: continue
            temp.append(temp_token[:])
            m.append(temp_mask)
            self.ll.append(len(temp_token))
           # print(srcs)
        print(len(temp))
        self.post = pad_sequence([torch.LongTensor(x) for x in temp], batch_first=True, padding_value=0)
        self.mask = pad_sequence([torch.LongTensor(x) for x in m], batch_first=True, padding_value=0)
    def __getitem__(self, index):

        return self.post[index], self.mask[index], self.ll[index]

    def __len__(self):
        return len(self.post)