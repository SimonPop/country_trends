import torch
from whale_dataset import WhaleDataset
from torch_geometric_temporal.nn import MTGNN
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from config import CONFIG
from pathlib import Path
import pandas as pd
torch.manual_seed(0)

def prepare_data():
    DATA_PATH = Path(__file__).parent.parent.parent / 'whales.csv'
    dataframe = pd.read_csv(DATA_PATH, parse_dates=['date']).set_index('date')
    train_dataframe = dataframe.iloc[:-len(dataframe)//10]
    val_dataframe = dataframe.iloc[-len(dataframe)//10:]
    
    train_dataset = WhaleDataset(dataframe = train_dataframe, x_input_length=CONFIG.model_config.seq_length - 1, y_input_length=30)
    val_dataset = WhaleDataset(dataframe = val_dataframe, x_input_length=CONFIG.model_config.seq_length - 1, y_input_length=30)
    
    train_loader = DataLoader(train_dataset, batch_size=32)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    return train_loader, val_loader, train_dataset, val_dataset

if __name__ == "__main__":
    train_loader, val_loader, train_dataset, val_dataset = prepare_data()

    A_superhighway = train_dataset.A_superhighway
    
    model_config = CONFIG.model_config
    model_config.num_nodes = len(train_dataset.dataframe.columns)

    model = MTGNN(
            **model_config.dict()
        )
    
    for parameter in model.parameters():
        print(parameter.size())

    criterion = torch.nn.L1Loss()
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=CONFIG.momentum)
    optimizer = optim.Adam(model.parameters())
    running_loss = 0

    for epoch in range(CONFIG.epochs):
        # Train
        for i, (x, y) in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()
            outputs = model(x)

            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f'[{epoch + 1}, {i + 1:5d}] train loss: {running_loss/len(train_loader):.3f}')
        running_loss = 0.0
        print(model._graph_constructor(model._idx, FE=None))
        # Validation
        for i, (x, y) in tqdm(enumerate(val_loader)):
            optimizer.zero_grad()
            outputs = model(x)

            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f'[{epoch + 1}, {i + 1:5d}] validation loss: {running_loss/len(val_loader):.3f}')
        running_loss = 0.0

    PATH = Path(__file__).parent.parent.parent / "artifacts" / "model.pt"
    torch.save(model.state_dict(), PATH)