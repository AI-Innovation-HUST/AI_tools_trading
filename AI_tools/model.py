from transformer import *
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import ast
from numpy import load
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import torch
import random
from tqdm import tqdm

def compute_acc(pred, ground_truth,threshold=0.08):
    diff_h = (torch.abs((pred[:,0] - ground_truth[:,0]) / ground_truth[:,0])).sum()/pred.size(0)
    diff_l =(torch.abs((pred[:,1] - ground_truth[:,1]) / ground_truth[:,1])).sum()/pred.size(0)

    return diff_h,diff_l

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



def evaluate(eval_model,epoch,criterion,data_source,dev='cpu'):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    ntokens = 28
    count = 0
    with torch.no_grad():
        cum_loss = 0
        acc_count = 0
        accs_h = 0
        accs_l = 0
        for batch in tqdm(data_source):
            data, targets = batch
            # targets = embs(targets.long())
            targets = targets.view(-1,2).to(dev)
            output = eval_model(data.to(dev))
            output = output.view(-1,2)
            loss = criterion(output,targets)
            score_h,score_l = compute_acc(output,targets)
            accs_h += score_h
            accs_l += score_l
            # accs += ((torch.argmax(output,dim=1)==targets).sum().item()/output.size(0))
            cum_loss += loss.item()
            count+=1
        print(epoch,"Loss: ",(cum_loss/count))
    return cum_loss/ (count), float(accs_h/(count)), float(accs_l/count)



    
   