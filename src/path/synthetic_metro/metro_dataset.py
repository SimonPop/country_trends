from cycling_graph import CyclingGraph
import torch
from typing import List
from pydantic import BaseModel

# FIXME: possible to generate more samples.

class Cycle(BaseModel):
    nodes: List[int] = [1,2,3,4] 
    weight: float = .2
    is_cycle: bool = False

class MetroDataset(torch.utils.data.Dataset):
    def __init__(self, cycles: List[Cycle], high: int = 100, n_steps: int = 100, init_nb: int = 5):
        self.high = high
        self.n_steps = n_steps
        self.init_nb = init_nb
        self.cg = CyclingGraph()
        self.init_graph(cycles)
        self.dataframes = [
            self.init_steps()
            for i in range(init_nb)
        ] # Creates multiple samples based on different initializations.

    def init_graph(self, cycles: List[Cycle]):
        for cycle in cycles:
            self.cg.add_line(cycle.nodes, cycle.weight, cycle.is_cycle)

    def init_steps(self):
        init_vector = self.cg.random_initialization(0, self.high)
        # init_vector = init_vector*0
        # init_vector[0] = 10
        X = self.cg.simulate(
            init_vector, self.n_steps
        )
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
    dataset = MetroDataset()
    print(dataset[0])
    
