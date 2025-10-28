from datetime import datetime
from config import InformerConfig
from exp.exp_informer import Exp_Informer
from utils.tools import dotdict

def train_informer(dataset_path: str, config: InformerConfig) -> dict:
    args = dotdict(config.model_dump() | {'data_path': dataset_path})
    
    exp = Exp_Informer(args)
    setting = f'llm_pl{config.pred_len}_{datetime.now().strftime("%H%M%S")}'
    exp.train(setting)
    mae_val, mse_val = exp.test(setting, flag='val')
    mae_test, mse_test = exp.test(setting, flag='test')
    
    return {'mae_val': mae_val, 'mse_val': mse_val, 'mae_test': mae_test, 'mse_test': mse_test}