import torch
from torch import optim, nn, utils, Tensor
from torch_geometric.nn import GraphConv
import pytorch_lightning as pl
from icecream import ic

class DummyModel(pl.LightningModule):
    def __init__(self, embedding_size: int, node_nb: int, neighbor_nb: int, input_size: int):
        super().__init__()
        self.neighbor_nb = neighbor_nb
        self.embedding_size = embedding_size
        self.node_nb = node_nb

        self.node_embeddings = torch.nn.Embedding(node_nb, embedding_size)
        self.graph_layer = GraphConv(input_size, 1)
        self.linear = nn.Linear(10, 10)

        self.softmax = nn.Softmax(dim=-1)

    def graph_learning(self, node_embeddings: torch.tensor) -> torch.tensor:
        # 1. Compute similarity between nodes.
        A = torch.mm(node_embeddings, node_embeddings.transpose(1, 0))
        # 2. Make sure only positive values can occur.
        A = torch.nn.functional.relu(A)
        # 3. Sample top-k neighbors.
        values, indices = A.topk(k=self.neighbor_nb+1, dim=1)
        mask = torch.zeros_like(A)
        mask.scatter_(1, indices, values.fill_(1))
        return A*mask

    def forward(self, X: torch.tensor):
        # 1. Get adjacency matrix.
        adj = self.graph_learning(self.node_embeddings.weight)
        edge_index = adj.nonzero().t().contiguous()
        # edge_index = torch.stack((torch.tensor(range(10)), torch.tensor([(i+1)%10 for i in range(10)])))
        ic(edge_index)
        # 2. Compute output.
        y = self.graph_layer(X, edge_index).squeeze(-1) # TODO: directement utilier la matrice pour commencer
        ic(y)
        # y_tmp = torch.bmm(adj, X.squeeze(-1).T)
        # y = self.linear(X.squeeze(-1))
        return y
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        print(y, y_hat)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-1)
        return optimizer
    
    def dummy_embeddings(self, n_nodes: int):
        sub = torch.cat((torch.eye(n_nodes-1), torch.zeros((n_nodes-1, 1))), dim=1)
        sub = torch.cat((torch.zeros((1, n_nodes)), sub), dim=0)
        embeddings = sub + sub.T + torch.eye(n_nodes)
        return embeddings
    

if __name__ == "__main__":
    from eye_dataset import EyeDataset
    from pytorch_lightning import Trainer
    import torch
    size=10
    dataset = EyeDataset(size=size)
    train_loader = torch.utils.data.DataLoader(dataset)
    trainer = Trainer(max_epochs=100)
    model = DummyModel(embedding_size=8, node_nb=size, neighbor_nb=1, input_size=1)
    trainer.fit(model, train_loader)

    # X, y = dataset[0]
    # ic(X)
    # ic(y)
    # y_hat = model.forward(X.unsqueeze(0))
    # ic(y_hat)
    # ic(torch.nn.functional.cross_entropy(y_hat, y.unsqueeze(0)))

    # torch.tensor([1,1,3,1])