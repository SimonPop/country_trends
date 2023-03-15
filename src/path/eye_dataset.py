import pandas as pd
import torch
import h3
import geopandas as gpd

class EyeDataset(torch.utils.data.Dataset):
    def __init__(self, size: int = 10):
        self.size = size
        self.eye = torch.eye(10).float()

    def __len__(self):
        return self.size - 1

    def __getitem__(self, index):
        x = self.eye[index].unsqueeze(-1)
        y = self.eye[index+1]
        return x, y
    
