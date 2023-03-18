from cycling_graph import CyclingGraph
import pandas as pd
import torch
from typing import List
from pydantic import BaseModel

# FIXME: possible to generate more samples.

class Cycle(BaseModel):
    nodes: List[int] = [1,2,3,4] 
    weight: float = .2
    is_cycle: bool = False

class MetroDataset(torch.utils.data.Dataset):
    def __init__(self, cycles: List[Cycle], high: int = 100, n_steps: int = 100):
        self.high = high
        self.n_steps = n_steps
        self.cg = CyclingGraph()
        self.init_graph(cycles)
        self.dataframe = self.init_steps()

    def init_graph(self, cycles: List[Cycle]):
        for cycle in cycles:
            self.cg.add_line(cycle.nodes, cycle.weight, cycle.is_cycle)
        # self.cg.add_line([3,6,7,8], .3, True)

    def init_steps(self):
        init_vector = self.cg.random_initialization(0, self.high)
        X = self.cg.simulate(
            init_vector, self.n_steps
        )
        return X

    def __len__(self):
        return self.n_steps-1

    def __getitem__(self, index):
        x = torch.FloatTensor(self.dataframe.iloc[index].values).unsqueeze(-1)
        y = torch.FloatTensor(self.dataframe.iloc[index+1].values)
        return x, y
    
if __name__ == "__main__":
    dataset = MetroDataset()
    print(dataset[0])
    
