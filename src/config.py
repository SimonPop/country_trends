from pydantic import BaseModel
from typing import List
from pathlib import Path
import yaml

class ModelConfig(BaseModel):
    gcn_true: bool
    build_adj: bool
    gcn_depth: int
    num_nodes: int
    seq_length: int
    kernel_set: List[int]
    kernel_size: int
    dropout: float 
    subgraph_size: int # Warning: need to be lower than num_nodes
    node_dim: int
    dilation_exponential: int 
    conv_channels: int
    residual_channels: int 
    skip_channels: int
    end_channels: int
    in_dim: int # Number of features per node (1 in our case) 
    out_dim: int #Correspond to the seq length in y
    layers: int 
    propalpha: float 
    tanhalpha: int
    layer_norm_affline: bool

class Config(BaseModel):
    model_config: ModelConfig    
    epochs: int
    momentum: float
    lr: float
    countries: List[str] = None

with open(Path(__file__).parent / "config.yaml") as f:
        CONFIG = Config(**yaml.safe_load(f))