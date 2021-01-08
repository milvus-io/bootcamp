from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import sys
import pickle
import numpy as np
import lmdb
import torch
from src.config import file_keys_pkl


       
class RecipeLoader(data.Dataset):
    def __init__(self, data_path=None):

        if data_path == None:
            raise Exception('No data path specified.')

        self.env = lmdb.open(data_path, max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)

        with open(file_keys_pkl, 'rb') as f:
            self.ids = pickle.load(f)

        self.maxInst = 20



    def __getitem__(self, index):
        recipId = self.ids[index]

        with self.env.begin(write=False) as txn:
            serialized_sample = txn.get(self.ids[index].encode('latin1'))
        sample = pickle.loads(serialized_sample,encoding='latin1')

        # instructions
        instrs = sample['intrs']
        itr_ln = len(instrs)
        t_inst = np.zeros((self.maxInst, np.shape(instrs)[1]), dtype=np.float32)
        t_inst[:itr_ln][:] = instrs
        instrs = torch.FloatTensor(t_inst)

        # ingredients
        ingrs = sample['ingrs'].astype(int)
        ingrs = torch.LongTensor(ingrs)
        igr_ln = max(np.nonzero(sample['ingrs'])[0]) + 1

        return [instrs, itr_ln, ingrs, igr_ln], recipId

    def __len__(self):
        return len(self.ids)
