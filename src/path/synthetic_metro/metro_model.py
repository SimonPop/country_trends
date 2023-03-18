import torch
from torch import optim, nn
from torch_geometric.nn import GraphConv
import pytorch_lightning as pl
from icecream import ic

class MetroModel(pl.LightningModule):
    def __init__(self, embedding_size: int, num_nodes: int, neighbor_nb: int, input_size: int, gsl_mode: str = "matrix"):
        super().__init__()
        self.neighbor_nb = neighbor_nb
        self.embedding_size = embedding_size
        self.num_nodes = num_nodes

        self.gsl_mode = gsl_mode

        self.node_embeddings_start = torch.nn.Embedding(num_nodes, embedding_size, sparse=False)
        self.node_embeddings_target = torch.nn.Embedding(num_nodes, embedding_size, sparse=False)
        self.graph_layer = GraphConv(input_size, 1)
        self._idx = torch.arange(self.num_nodes)
        self._linear1 = nn.Linear(embedding_size, embedding_size)
        self._linear2 = nn.Linear(embedding_size, embedding_size)

        self.matrix = nn.Parameter(torch.randn(num_nodes, num_nodes), requires_grad=True)

        self._alpha = 0.1
        self.softmax = nn.Softmax(dim=-1)

    def graph_matrix_learning(self) -> torch.tensor:
        # A = torch.nn.functional.relu(self.matrix)
        A = self.matrix # if not used with topk, use that instead .exp()
        A = A.exp()
        dim=0
        values, indices = A.topk(k=self.neighbor_nb+1, dim=dim)
        mask = torch.zeros_like(A)
        mask.scatter_(dim, indices, values.fill_(1))
        return A*mask
    
    def graph_emb_learning(self) -> torch.tensor:
        nodevec1 = self.node_embeddings_start(self._idx)
        nodevec2 = self.node_embeddings_target(self._idx)

        nodevec1 = torch.tanh(self._alpha *self._linear1(nodevec1)) 
        nodevec2 = torch.tanh(self._alpha *self._linear2(nodevec2))

        A = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(
            nodevec2, nodevec1.transpose(1, 0)
        )
        # Topk on dim=1: A node can only have k sources 
        # Topk on dim=0: A node can only have k targets 
        dim=1 # FIXME: 1 or 0 ? 
        values, indices = A.topk(k=self.neighbor_nb+1, dim=dim)
        mask = torch.zeros_like(A)
        mask.scatter_(dim, indices, values.fill_(1))
        return A*mask

    def graph_learning(self) -> torch.tensor:
        if self.gsl_mode == "matrix":
            return self.graph_matrix_learning()
        elif self.gsl_mode == "embedding":
            return self.graph_emb_learning()
        else:
            raise ValueError('Unkown mode.')
    
    def simple_graph_mult(self, A, X):
        y = torch.einsum("nwl,vw->nvl", (X, A))
        y = y.squeeze(-1)
        return y
    
    def gnn(self, A, X):
        edge_index = A.nonzero().t().contiguous()
        y = self.graph_layer(X, edge_index).squeeze(-1) # TODO: directement utilier la matrice pour commencer
        return y
    
    def forward(self, X: torch.tensor):
        A = self.graph_learning()
        y = self.simple_graph_mult(A, X)
        # y = self.linear(X)
        return y
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x) #.squeeze(-1)
        # loss = torch.nn.functional.cross_entropy(y_hat, y)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-2)
        # optimizer = torch.optim.SGD(self.parameters(), lr=0.0001)
        return optimizer
    
    def dummy_embeddings(self, n_nodes: int):
        sub = torch.cat((torch.eye(n_nodes-1), torch.zeros((n_nodes-1, 1))), dim=1)
        sub = torch.cat((torch.zeros((1, n_nodes)), sub), dim=0)
        embeddings = sub + sub.T + torch.eye(n_nodes)
        return embeddings