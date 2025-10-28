from dataclasses import dataclass
from pydantic import BaseModel, create_model
from typing import List, Dict, Any, Union
import os, torch
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GRIDS = {
    'grid12': {
        'seq_len': [24, 48, 96, 168, 336, 720],
        'label_len': [24, 48, 96, 168, 336, 720],
        'e_layers': [2, 3, 4, 6],
        'n_heads': [8, 16],
        'factor': [3, 5, 8, 10],
    },
    'grid3': {
        'seq_len': [24, 48, 96, 168, 336, 720],
        'label_len': [24, 48, 96, 168, 336, 720],
        'e_layers': [2, 3, 4, 6],
        'n_heads': [8, 16],
        'factor': [3, 5, 8, 10],
        'd_model': [256, 512, 768],
        'd_ff': [1024, 2048, 3072],
        'd_layers': [1, 2, 3],
        'dropout': [0.05, 0.1, 0.2],
    }
}

PARAM_DESCRIPTIONS = {
    'seq_len': 'Input sequence length of Informer encoder',
    'label_len': 'Start token length of Informer decoder',
    'pred_len': 'Prediction sequence length',
    'd_model': 'Dimension of model',
    'n_heads': 'Num of heads',
    'e_layers': 'Num of encoder layers',
    'd_layers': 'Num of decoder layers',
    'd_ff': 'Dimension of fcn',
    'factor': 'Probsparse attn factor',
    'dropout': 'The probability of dropout',
}

PRED_LENS = [24, 48, 168, 336, 720]
JOURNAL_DIR = './experiments'

class InformerConfig(BaseModel):
    # Динамические параметры (оптимизируемые)
    hypothesis: str = ""
    seq_len: int = 96
    label_len: int = 48
    pred_len: int = 24
    e_layers: int = 2
    n_heads: int = 8
    factor: int = 5
    d_model: int = 512
    d_ff: int = 2048
    d_layers: int = 1
    dropout: float = 0.05
    
    # Константы обучения
    model: str = 'informer'
    data: str = 'custom'
    root_path: str = './'
    features: str = 'MS'
    target: str = 'OT'
    freq: str = 'h'
    checkpoints: str = './informer_checkpoints'
    attn: str = 'prob'
    embed: str = 'timeF'
    activation: str = 'gelu'
    distil: bool = True
    output_attention: bool = False
    mix: bool = True
    padding: int = 0
    batch_size: int = 32
    learning_rate: float = 0.0001
    loss: str = 'mse'
    lradj: str = 'type1'
    train_epochs: int = 6
    patience: int = 3
    num_workers: int = 0
    itr: int = 1
    des: str = 'llm_search'
    enc_in: int = 7
    dec_in: int = 7
    c_out: int = 1
    detail_freq: str = 'h'

    # GPU конфигурация
    use_gpu: bool = torch.cuda.is_available()
    use_multi_gpu: bool = torch.cuda.device_count() > 1
    device_ids: list = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    devices: str = ','.join(map(str, range(torch.cuda.device_count()))) if torch.cuda.is_available() else ''
    gpu: int | None = 0 if torch.cuda.is_available() else None
    use_amp: bool = False

def build_llm_config(grid: Dict[str, List]) -> type[BaseModel]:
    fields = {name: (type(values[0]), ...) for name, values in grid.items()}
    fields['hypothesis'] = (str, ...)
    return create_model('LLMConfig', **fields)

def build_llm_list_config(grid: Dict[str, List]) -> type[BaseModel]:
    llm_config = build_llm_config(grid)
    return create_model('ListConfigs', list_of_configs=(List[llm_config], ...))

def get_train_config(updates: InformerConfig, pred_len: int = 24) -> InformerConfig:
    base_config = InformerConfig(pred_len=pred_len)
    extra_fields = set(updates.model_dump()) - set(type(base_config).model_fields)
    if extra_fields: raise ValueError(f"Лишние поля в cfg: {extra_fields}")
    train_cfg = base_config.model_copy(update=updates.model_dump())
    return train_cfg