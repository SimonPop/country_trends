from metro_dataset import MetroDataset, Cycle
from pytorch_lightning import Trainer
from metro_model import MetroModel
from icecream import ic
import torch

cycle_1 = Cycle(nodes=list(range(5)), weight=0.2, cycle=True)
# cycle_2 = Cycle([1,2,3,4], 0.2, False)

dataset = MetroDataset([cycle_1])
num_nodes = dataset.cg.num_stations()

ic(dataset.cg.adjacency_matrix())

train_loader = torch.utils.data.DataLoader(dataset, batch_size=1)
trainer = Trainer(max_epochs=50)
model = MetroModel(embedding_size=8, num_nodes=num_nodes, neighbor_nb=2, input_size=1, gsl_mode="matrix")
trainer.fit(model, train_loader)

for i in range(len(dataset)):
    X,y = dataset[i]
    X = X.unsqueeze(0)
    y = y.unsqueeze(0)
    # ic(y, model(X))

ic(model.graph_matrix_learning())

X,y = dataset[3]
X = X.unsqueeze(0)
y = y.unsqueeze(0)
ic(X, y, model(X))


# L'addition du nombre de personnes par quai à chaque station rend impossible la création d'un graphe exact puisque l'on ne connaît pas la proportion qui va partir d'un côté ou de l'autre seulement en regardant la somme totale, mais en regardant le nombre par quais.