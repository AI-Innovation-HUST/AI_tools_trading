from transformer import *
from essential_functions import *
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import ast
from numpy import load
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import torch
import random
from coin_dataloader import *
from tqdm import tqdm

def compute_acc(pred, ground_truth,threshold=0.04):
    diff = (pred - ground_truth) / ground_truth
    score = torch.where(diff < threshold, 1.0, 0.0)
    return score

class Transformer(nn.Module):
    def __init__(self,n_blocks=2,d_model=16,n_heads=4,d_ff=256,dropout=0.2,vocab_size=256):
        super().__init__()
        self.emb = WordPositionEmbedding(vocab_size = vocab_size,d_model=d_model)
        self.decoder_emb = WordPositionEmbedding(vocab_size=vocab_size,d_model=d_model)
        self.encoder = TransformerEncoder(n_blocks=n_blocks,d_model=d_model,n_heads=n_heads,d_ff=d_ff,dropout=0.2)
        self.decoder = TransformerDecoder(n_blocks=n_blocks,d_model=d_model,d_feature=16,n_heads=n_heads,d_ff=d_ff,dropout=0.2)
    
    def forward(self,x):
        g = self.emb(x)
        encoded = self.encoder(g)
        # print("Pass encoder: ",encoded.shape)
        p = self.decoder_emb(x)
        # print("pass decoder",p.shape)
        y = self.decoder(p, encoded)
        # print("SHAPE",y.shape)
        return y;



def evaluate(eval_model,epoch,criterion,data_source):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    ntokens = 28
    count = 0
    with torch.no_grad():
        cum_loss = 0
        acc_count = 0
        accs = 0
        for batch in tqdm(data_source):
            data, targets = batch
            # targets = embs(targets.long())
            targets = targets.view(-1,2).to("cuda")
            output = eval_model(data.to("cuda"))
            output = output.view(-1,2)
            loss = criterion(output,targets)
            accs += (compute_acc(output,targets)).sum()/(2*(output.size(0)))
            # accs += ((torch.argmax(output,dim=1)==targets).sum().item()/output.size(0))
            cum_loss += loss.item()
            count+=1
        print(epoch,"Loss: ",(cum_loss/count))
    return cum_loss/ (count), accs/(count)



    
   