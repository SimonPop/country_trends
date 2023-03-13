import torch
from torch_geometric.nn import GCNConv
import pytorch_lightning as pl

class DummyModel(pl.LightningModule):
    def __init__(self, embedding_size: int, node_nb: int, neighbor_nb: int):
        super().__init__()
        self.neighbor_nb = neighbor_nb
        self.embedding_size = embedding_size
        self.node_nb = node_nb

        self.node_embeddings = torch.nn.Embedding(node_nb, embedding_size)
        self.grah_layer = GCNConv(embedding_size, 1)

    def graph_learning(self, node_embeddings: torch.tensor) -> torch.tensor:
        # 1. Compute similarity between nodes.
        A = torch.mm(node_embeddings, node_embeddings.transpose(1, 0))
        # 2. Make sure only positive values can occur.
        A = torch.nn.functional.relu(A)
        # 3. Sample top-k neighbors.
        values, indices = A.topk(k=self.neighbor_nb, dim=1)
        mask = torch.zeros_like(A)
        mask.scatter_(1, indices, values.fill_(1))
        return A*mask

    def forward(self, X: torch.tensor):
        # 1. Get adjacency matrix.
        adj = self.graph_learning(self.node_embeddings.weight)
        print(adj.size())
        edge_index = adj.nonzero().t().contiguous()
        print(edge_index.size())
        # 2. Compute output.
        print(X.size())
        y = self.grah_layer(X, edge_index)
        return y