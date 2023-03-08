import torch
from whale_dataset import WhaleDataset
from torch_geometric_temporal.nn import MTGNN
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from config import CONFIG
from pathlib import Path

if __name__ == "__main__":

    dataset = WhaleDataset(x_input_length=CONFIG.model_config.seq_length - 1, y_input_length=3)
    loader = DataLoader(dataset, batch_size=8)

    A_tilde = dataset.A_pathway

    model_config = CONFIG.model_config
    model_config.num_nodes = len(dataset.dataframe.columns)

    model = MTGNN(
            **model_config.dict()
        )
    
    criterion = torch.nn.L1Loss()
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=CONFIG.momentum)
    optimizer = optim.Adam(model.parameters())
    running_loss = 0

    for epoch in range(CONFIG.epochs):
        for i, (x, y) in enumerate(loader):
            optimizer.zero_grad()
            outputs = model(x, A_tilde=A_tilde)

            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss/len(loader):.3f}')
        running_loss = 0.0

    PATH = Path(__file__).parent.parent / "artifacts" / "model.pt"
    torch.save(model.state_dict(), PATH)