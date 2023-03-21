from pydantic import BaseModel
from typing import List, Any
from pathlib import Path
import yaml

class Line(BaseModel):
    weight: float
    nodes: List[Any]

class Config(BaseModel):
    lines: List[Line]


with open(Path(__file__).parent / "config/topo_1.yaml", 'r') as f:
    CONFIG = yaml.safe_load(f)
    CONFIG = Config(**CONFIG)
