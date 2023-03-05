import torch
from dataset import Dataset
from torch_geometric_temporal.nn import MTGNN
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

if __name__ == "__main__":

    dataset = Dataset()
    loader = DataLoader(dataset, batch_size=8)

    model = MTGNN(
        gcn_true=True,
        build_adj=True,
        gcn_depth=2,
        num_nodes=5,
        seq_length=dataset.input_length + 1,
        kernel_set=[1], 
        kernel_size=1, 
        dropout=0.3, 
        subgraph_size=2, # Warning: need to be lower than num_nodes
        node_dim=40, 
        dilation_exponential=1, 
        conv_channels=32, 
        residual_channels=32, 
        skip_channels=64, 
        end_channels=128, 
        in_dim=1, # Number of features per node (1 in our case) 
        out_dim=9, #Correspond to the seq length in y
        layers=3, 
        propalpha=0.05, 
        tanhalpha=3,
        layer_norm_affline=True
        )

    criterion = torch.nn.L1Loss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    running_loss = 0

    for epoch in range(2):
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