from data.eye_dataset import EyeDataset
from pytorch_lightning import Trainer
from model.dummy_model import DummyModel
from icecream import ic
import torch

size=10
dataset = EyeDataset(size=size)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=1)
trainer = Trainer(max_epochs=500)
model = DummyModel(embedding_size=8, num_nodes=size, neighbor_nb=1, input_size=1, gsl_mode="matrix")
trainer.fit(model, train_loader)

for i in range(len(dataset)):
    X,y = dataset[i]
    X = X.unsqueeze(0)
    y = y.unsqueeze(0)
    ic(y, model(X))