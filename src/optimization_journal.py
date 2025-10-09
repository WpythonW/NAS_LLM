import os
import json
import pandas as pd
from datetime import datetime
import numpy as np
from config import JOURNAL_DIR

class Journal:
    def __init__(self, filename=None):
        os.makedirs(JOURNAL_DIR, exist_ok=True)
        
        if filename is None:
            filename = f'experiments_pred_{datetime.datetime.now()}.json'
        
        self.filepath = os.path.join(JOURNAL_DIR, filename)
        self.entries = self._load()
    
    def _load(self):
        if not os.path.exists(self.filepath):
            return []
        
        with open(self.filepath, 'r') as f:
            content = f.read().strip()
            if not content:
                return []
            return json.loads(content)
    
    def _save(self):
        with open(self.filepath, 'w') as f:
            json.dump(self.entries, f, indent=2)
    
    def add(self, config, mse: float, mae: float, trial: int):
        if config.label_len >= config.seq_len:
            raise ValueError(f"label_len >= seq_len: {config.label_len} >= {config.seq_len}")
        
        entry = {
            'trial': int(trial),
            'timestamp': datetime.now().isoformat(),
            'mse': float(mse),
            'mae': float(mae),
            **{k: int(v) if isinstance(v, (np.integer, np.int64)) else v 
               for k, v in config.__dict__.items()}
        }
        self.entries.append(entry)
        self._save()
    
    def get_history_table(self, last_n=1500) -> str:
        if not self.entries:
            return ""
        
        df = pd.DataFrame(self.entries[-last_n:])[['seq_len', 'label_len', 'e_layers', 'n_heads', 'factor', 'mse', 'mae']]
        return "\n" + df.to_markdown(index=False, floatfmt=".4f") + "\n"
    
    def count_trials(self) -> int:
        return len(self.entries)
    
    def get_best(self):
        if not self.entries:
            return None
        df = pd.DataFrame(self.entries)
        return df.loc[df['mse'].idxmin()]
    
    def print_best(self):
        best = self.get_best()
        if best is None:
            return
        
        print(f"\n>>>MSE={best['mse']:.4f} MAE={best['mae']:.4f} "
              f"seq={int(best['seq_len'])} lbl={int(best['label_len'])} e={int(best['e_layers'])} "
              f"h={int(best['n_heads'])} f={int(best['factor'])}\n")