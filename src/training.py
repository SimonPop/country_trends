import torch
from dataset import Dataset
from torch_geometric_temporal.nn import MTGNN
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from config import CONFIG
from pathlib import Path

if __name__ == "__main__":

    dataset = Dataset(CONFIG.model_config.seq_length - 1, countries=CONFIG.countries)
    loader = DataLoader(dataset, batch_size=8)

    model_config = CONFIG.model_config
    model_config.num_nodes = len(dataset.dataframe.columns)

    model = MTGNN(
            **model_config.dict()
        )

    criterion = torch.nn.L1Loss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=CONFIG.momentum)
    running_loss = 0

    for epoch in range(CONFIG.epochs):
        for i, (x, y) in tqdm(enumerate(loader)):
            optimizer.zero_grad()
            outputs = model(x)

            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    PATH = Path(__file__).parent.parent / "artifacts" / "model.pt"
    torch.save(model.state_dict(), PATH)