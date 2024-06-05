import random
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import ast
from numpy import load
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import torch
import random

def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data

bptt = 2048
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader



class CustomDataLoader:
    def __init__(self,source):
        self.batches = list(range(len(source) - bptt))
        self.batches = random.shuffle(self.batches)
        self.data = source
        
    def get_batch(self,source,i):
        ind = self.batches[i]
        seq_len = min(bptt,len(self.data)-1-ind)
        src = self.data[ind:ind+seq_len]
        tar = self.data[ind+1:ind+1+seq_len].view(-1)
        return src,tar
        if(i==len(self.batches)-1):
            self.batches = random.shuffle(self.batches)
        return src,tar

def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def plot_multiple(data,legend):
    fig,ax = plt.subplots()
    for line in data:
        print(line)
        plt.plot(list(range(len(line))),line)
    plt.legend(legend)
    plt.show()

def plot_subplots(data,legends):
    names = ['Accuracy', 'Loss']
    print(data)
    plt.figure(figsize=(10, 5))
    for i in range(len(data)):
        # plt.subplot(200+i)
        plt.subplot(121+i)
        plt.plot(list(range(0,len(data[i])*50,50)),data[i])
        plt.title(legends[i])
        plt.xlabel("Epochs")
    plt.show()