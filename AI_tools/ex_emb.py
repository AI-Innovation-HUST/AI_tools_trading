import torch.nn as nn
import torch

i = torch.LongTensor([[1,2,3,4,5,6]])

print(i.shape)

layer1 = nn.Embedding(10,5)


out = layer1(i)

print(out)