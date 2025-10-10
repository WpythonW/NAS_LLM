from dataclasses import dataclass
from pydantic import BaseModel
from typing import List
import os
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

@dataclass
class Config(BaseModel):
    hypothesis: str
    seq_len: int
    label_len: int
    e_layers: int
    n_heads: int
    factor: int

@dataclass
class ListConfigs(BaseModel):
    lisf_of_configs: List[Config]

@dataclass
class InformerConfig:
    hypothesis: str
    seq_len: int
    label_len: int
    e_layers: int
    n_heads: int
    factor: int

GRID = {
    'seq_len': [24, 48, 96, 168, 336, 720],
    'label_len': [24, 48, 96, 168, 336, 720],
    'e_layers': [2, 3, 4, 6],
    'n_heads': [8, 16],
    'factor': [3, 5, 8, 10],
}

FIXED = {
    'd_model': 512,
    'd_ff': 2048,
    'd_layers': 2,
    'dropout': 0.05,
}

PRED_LENS = [24, 48, 168, 336, 720]
JOURNAL_DIR = './experiments'