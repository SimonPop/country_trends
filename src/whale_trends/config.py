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


COUNTRIES = ['AR', 'BR', 'CA', 'US', 'MX', 'AU', 'NZ', 'ZA', 'FR', 'GB', 'IE', 'PT', 'NO', 'LK', 'IS', 'ID', 'ES']
if not CONFIG.countries is None:
    COUNTRIES = [c for c in COUNTRIES if c in CONFIG.countries]
REGIONS = [] # TODO

GEOPOINT = {
     'AR': (-38.08287775524158, -64.23765667336085), 
     'BR': (-9.882452900896004, -45.39678184345796), 
     'CA': (54.12195962944559, -102.19049864545875), 
     'US': (40.33582787278787, -99.886218990003), 
     'MX': (20.773077645789222, -100.0217649003884), 
     'AU': (-28.54320059040562, 135.90846378443294), 
     'NZ': (-43.27964630810205, 170.6082044783548), 
     'ZA': (-31.12957903397068, 25.30304032255698), 
     'FR': (48.28164965906673, 2.2231198002255055), 
     'GB': (53.07943434560134, -1.6880130574100323), 
     'IE': (52.65498722796102, -7.796411366872499), 
     'PT': (40.49383062236667, -7.441201024283314), 
     'NO': (61.898901052399765, 9.20139584198989), 
     'LK': (7.103827811336456, 80.90515823200094), 
     'IS': (64.76658471679751, -17.745425185304367), 
     'ID': (-7.495383850582282, 109.50533513206938), 
     'ES': (40.22594338099514, -3.288369000402384)
}

SUPERHIGHWAY =  {
     'AR': 0, 
     'BR': 1, 
     'ZA': 1, 
     'CA': 2, 
     'US': 2, 
     'MX': 2, 
     'AU': 3, 
     'NZ': 3, 
     'ID': 3, 
     'FR': 5, 
     'GB': 5, 
     'IE': 5, 
     'PT': 5, 
     'NO': 5,
     'IS': 5, 
     'ES': 5,
     'LK': 6
}

