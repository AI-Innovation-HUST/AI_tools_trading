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
class Transformer(nn.Module):
    def __init__(self,n_blocks=2,d_model=16,n_heads=4,d_ff=256,dropout=0.2,vocab_size=128):
        super().__init__()
        self.emb = WordPositionEmbedding(vocab_size = vocab_size,d_model=d_model)
        self.decoder_emb = WordPositionEmbedding(vocab_size=vocab_size,d_model=d_model)
        self.encoder = TransformerEncoder(n_blocks=n_blocks,d_model=d_model,n_heads=n_heads,d_ff=d_ff,dropout=0.2)
        self.decoder = TransformerDecoder(n_blocks=n_blocks,d_model=d_model,d_feature=16,n_heads=n_heads,d_ff=d_ff,dropout=0.2)
    
    def forward(self,x):
        g = self.emb(x.unsqueeze(0))
        encoded = self.encoder(g)
        # print("Pass encoder: ",encoded.shape)/\
        p = self.decoder_emb(x)
        # print("pass decoder")
        y = self.decoder(p, encoded)
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
        for batch, i in enumerate(data_source):
            data, targets = data_source
            # targets = embs(targets)
            output = eval_model(data)
            output = output.view(-1,ntokens)
            loss = criterion(output,targets)
            accs += ((torch.argmax(output,dim=1)==targets).sum().item()/output.size(0))
            cum_loss += loss
            count+=1
        print(epoch,"Loss: ",(cum_loss/count),"Accuracy ",accs/count)
    return cum_loss/ (count)



    
   