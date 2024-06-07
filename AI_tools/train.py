import torch
import torch.nn as nn
from model import *
from dataloader_v2 import *
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import talib as ta

def get_data(files,ptype):
    data = []
    for file in files:
        f = open(file)
        pair = pd.read_csv(f)
        data.extend(list(pair[ptype]))
        f.close()
    return data
def calculate(df):
    # Select numerical column
    numerical_data = df['o'].to_numpy()
    mean = np.mean(numerical_data)
    std = np.std(numerical_data)
    # Print test results
    print("mean:",np.mean(numerical_data))
    print("std:", np.std(numerical_data))
    return mean,std
def load_data(csv_file, test_size=0.1, val_size=0.2):
    df = pd.read_csv(csv_file)
    df['SMA'] = ta.SMA(df['c'], timeperiod = 20)
    df['+DMI'] = ta.PLUS_DI(df['h'],df['l'],df['c'],timeperiod=14)
    df['-DMI'] = ta.MINUS_DI(df['h'],df['l'],df['c'],timeperiod=14)
    df['ADX'] = ta.ADX(df['h'],df['l'],df['c'],timeperiod=14)
    df['RSI'] = ta.RSI(df['c'],14)
    df['up_band'], df['mid_band'], df['low_band'] = ta.BBANDS(df['c'], timeperiod =20)
    mean,std = calculate(df)
    train_val_df, test_df = train_test_split(df, test_size=test_size, shuffle=False, random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=0.2, shuffle=False, random_state=42)

    train_dataset = CoinDataset(train_df,mean=mean,std=std)
    val_dataset = CoinDataset(val_df.reset_index(drop=True),mean=mean,std=std)
    test_dataset = CoinDataset(test_df.reset_index(drop=True),mean=mean,std=std)

    return train_dataset, val_dataset, test_dataset

if __name__ == '__main__':
    data = []
    dev = torch.device("cpu")
    train_dataset, val_dataset, test_dataset = load_data(csv_file='Dataset/test.csv')
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=1)
    print("Train dataset size:", len(train_dataset))
    print("Validation dataset size:", len(val_dataset))
    print("Test dataset size:", len(test_dataset))

    # procsd_data = load("Eavg_open.npy")
    # procsd_data = get_data(["Dataset/test.csv"],"o")
    # train_data =torch.tensor(procsd_data)[:30000*2]
    # print(train_data.shape)
    
    # val_data = torch.tensor(procsd_data)[30000*2:35000*2]
    # test_data = torch.tensor(procsd_data)[35000*2:]
    # train_data = train_data.to(dev)
    # val_data = val_data.to(dev)
    # test_data = test_data.to(dev)
    embs = torch.nn.Embedding(16,28)
    # batch_size = 16
    # ntokens = 28
    # train_data = batchify(train_data,batch_size)
    # # print(train_data.shape)
    # val_data = batchify(val_data,batch_size)
    # test_data = batchify(train_data,batch_size)
    model = Transformer(n_blocks=4,d_model=16,n_heads=8,d_ff=256,dropout=0.5)
    # model = torch.load("modelb1024")
    model.to(dev)
    
    criterion = nn.L1Loss()
    lr = 0.00001 # learning rate
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    accuracies = []
    lossies = []
    val_loss = []
    epochs = 1000
    for epoch in range(epochs):
        count = 0
        cum_loss = 0
        acc_count = 0
        accs = 0
        for batch in tqdm(train_loader):
            data, targets = batch
            targets = targets.view(-1,2)
            output = model(data)
            output = output.view(-1,2)
            # print(output.shape)
            loss = criterion(output.float(),targets.float())
            # rev = ((output - embs.weight).abs().sum(1)).nonzero()
            # print("rev",rev)

            # print(loss)
            # accs += ((torch.argmax(output,dim=1)==targets).sum().item()/output.size(0))
            cum_loss += loss
            loss.backward()
            optimizer.step()
            model.zero_grad()
            optimizer.zero_grad()
            count+=1
        print(epoch,"Loss: ",(cum_loss/count).item())
        if(epoch%50==1):
            # lossies.append(cum_loss.detach().numpy()/count)
            # accuracies.append(accs/count)
            # legend = ["accuracy","Loss"]
            # plot_subplots([accuracies,lossies],legend)
            # print("Valdata",val_loader.shape)
            print("------START EVAL------")
            eval_loss = evaluate(model,epoch,criterion,val_loader)
            print(epoch,"Loss: ",(cum_loss/count).item()," Valid_loss: ",eval_loss)
            if len(val_loss)>0 and eval_loss < val_loss[-1]:
                val_loss.append(eval_loss)
                torch.save(model,"evalModel.pth")
            else:
                val_loss.append(eval_loss)
                torch.save(model,"evalModel.pth")

        if(epoch%200)==0:
            torch.save(model,"modela.pth")