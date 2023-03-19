from cycling_graph import CyclingGraph
from metro_graph import MetroGraph
import torch
from typing import List
from pydantic import BaseModel
import pandas as pd

class Line(BaseModel):
    nodes: List[int] = [1,2,3,4] 
    weight: float = .2
    is_cycle: bool = False

class MetroDataset(torch.utils.data.Dataset):
    def __init__(self, lines: List[Line], high: int = 100, n_steps: int = 100, init_nb: int = 5):
        self.high = high
        self.n_steps = n_steps
        self.init_nb = init_nb
        self.cg = MetroGraph()
        self.init_graph(lines)
        self.dataframes = [
            self.init_steps()
            for _ in range(init_nb)
        ] # Creates multiple samples based on different initializations.

    def init_graph(self, lines: List[Line]):
        for line in lines:
            self.cg.add_line(line.nodes, line.weight)

    def init_steps(self):
        init_vector = self.cg.random_initialization(0, self.high)
        X = self.cg.simulate(
            init_vector, self.n_steps
        )
        X = pd.DataFrame(X)
        return X

    def __len__(self):
        return (self.n_steps-1)*self.init_nb

    def __getitem__(self, index):
        dataframe = self.dataframes[index%self.init_nb]
        index = index // self.init_nb
        x = torch.FloatTensor(dataframe.iloc[index].values).unsqueeze(-1)
        y = torch.FloatTensor(dataframe.iloc[index+1].values)
        return x, y

if __name__ == "__main__":
    dataset = MetroDataset([Line(nodes=[1,2,3,4,5], weight=0.2)])
    
    print(dataset[0])
    
