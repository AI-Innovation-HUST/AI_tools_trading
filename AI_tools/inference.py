from model import *
import torch




if __name__ =="__main__":
    model_path = ''
    dev = torch.device("cpu")

    model = Transformer(n_blocks=4,d_model=16,n_heads=8,d_ff=256,dropout=0.5)
    model.to(dev)

    model.load_state_dict(torch.load(model_path))
    
