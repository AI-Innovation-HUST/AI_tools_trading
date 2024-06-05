import random
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader,Dataset
import ast
from numpy import load
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import torch
import random
import pandas as pd
class CoinDataset(Dataset):
    def __init__(self, data_frame, transform=None):
        self.data_frame = data_frame
        # print("Size of dataframe: ",len(self.data_frame['o']))
        self.transform = transform
        self.sample = None 
        self.target = None

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        if idx < len(self.data_frame['o'])-1:
            self.sample = self.data_frame['o'][idx]/100000
            self.target = self.data_frame['o'][idx+1]/100000
            # sample = {col: sample[col] for col in self.data_frame.columns}

            if self.transform:
                self.sample = self.transform(self.sample)

        return self.sample,self.target

# Ví dụ về cách sử dụng CoinDataset và DataLoader
# if __name__ == "__main__":
#     coin_dataset = CoinDataset(csv_file='Dataset/test.csv')

#     dataloader = DataLoader(coin_dataset, batch_size=16, shuffle=True, num_workers=0)

#     for i_batch, sample_batched in enumerate(dataloader):
#         print(i_batch, sample_batched)