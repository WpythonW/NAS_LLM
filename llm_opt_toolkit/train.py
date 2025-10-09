import numpy as np
from datetime import datetime
from config import InformerConfig, FIXED
from exp.exp_informer import Exp_Informer
from utils.tools import dotdict
import torch

def train_informer(dataset_path, config: InformerConfig, pred_len: int) -> tuple[float, float]:
    """Обучение Informer и возврат MSE, MAE"""
    
    args = dotdict()
    
    # GPU конфигурация
    has_gpu = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if has_gpu else 0
    
    args.use_gpu = has_gpu
    args.use_multi_gpu = gpu_count > 1
    args.device_ids = list(range(gpu_count)) if has_gpu else []
    args.devices = ','.join(map(str, args.device_ids)) if has_gpu else ''
    args.gpu = 0 if has_gpu else None
    args.use_amp = False
    
    # Базовые параметры
    args.model = 'informer'
    args.data = 'custom'
    args.root_path = './'
    args.data_path = dataset_path
    args.features = 'MS'
    args.target = 'OT'
    args.freq = 'h'
    args.checkpoints = './informer_checkpoints'
    args.attn = 'prob'
    args.embed = 'timeF'
    args.activation = 'gelu'
    args.distil = True
    args.output_attention = False
    args.mix = True
    args.padding = 0
    args.batch_size = 32
    args.learning_rate = 0.0001
    args.loss = 'mse'
    args.lradj = 'type1'
    args.train_epochs = 6
    args.patience = 3
    args.num_workers = 0
    args.itr = 1
    args.des = 'llm_search'
    
    # Параметры модели
    args.d_model = FIXED['d_model']
    args.d_ff = FIXED['d_ff']
    args.d_layers = FIXED['d_layers']
    args.dropout = FIXED['dropout']
    args.seq_len = config.seq_len
    args.label_len = config.label_len
    args.pred_len = pred_len
    args.e_layers = config.e_layers
    args.n_heads = config.n_heads
    args.factor = config.factor
    args.enc_in = args.dec_in = 7
    args.c_out = 1
    args.detail_freq = args.freq
    args.freq = args.freq[-1:]
    
    exp = Exp_Informer(args)
    setting = f'llm_pl{pred_len}_{datetime.now().strftime("%H%M%S")}'
    exp.train(setting)
    exp.test(setting)
    
    metrics = np.load(f'./results/{setting}/metrics.npy')
    return float(metrics[0]), float(metrics[1])