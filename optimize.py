from datetime import datetime
import argparse
import numpy as np
import pandas as pd
from src.optimization_journal import Journal
from src.llm_requests import call_llm
from src.train import train_informer
from src.prompting import build_prompt

def print_comparison_table(results):
    # Фильтруем и создаём DataFrame
    df = pd.DataFrame([r for r in results if r])
    
    # Формируем строку конфигурации
    df['config'] = df[['seq_len', 'label_len', 'e_layers', 'n_heads', 'factor']].astype(int).astype(str).agg('/'.join, axis=1)
    
    # Выбираем колонки и сортируем
    df = df[['pred_len', 'mse', 'mae', 'config']].sort_values('pred_len')
    
    # Выводим таблицу
    print(f"\n{'='*80}\nРЕЗУЛЬТАТЫ\n{'='*80}")
    print(df.to_markdown(index=False, floatfmt=".4f"))
    print(f"{'='*80}\n")


def run_experiment(dataset_path, pred_len: int, journal_name: str, n_batches: int = 3, batch_size: int = 5):
    print(f"\n{'='*60}\nPRED_LEN={pred_len}ч | {datetime.now()}\n{'='*60}\n")
    
    journal = Journal(filename=journal_name)
    start_trial = journal.count_trials()
    total = n_batches * batch_size
    trials = np.arange(start_trial + 1, start_trial + total + 1)
    
    for idx, trial_num in enumerate(trials):
        #batch = idx // batch_size
        
        if idx % batch_size == 0:
            prompt = build_prompt(pred_len, batch_size, journal.get_history_table())
            #print(prompt)
            configs = call_llm(prompt)
        
        cfg = configs[idx % batch_size]
        print(f"[{trial_num}/{start_trial + total}] seq={cfg.seq_len} lbl={cfg.label_len} e={cfg.e_layers} h={cfg.n_heads} f={cfg.factor} ", end='')
        
        mse, mae = train_informer(dataset_path, cfg, pred_len)
        journal.add(cfg, mse, mae, trial_num)
        
        print(f"→ MSE={mse:.4f} MAE={mae:.4f}")
        
        if (idx + 1) % batch_size == 0:
            print(f"\n{'-'*60}")
    
    journal.print_best()
    return journal

#!rm ./journal/exp.json
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='Путь к датасету')
    parser.add_argument('--pred_len', type=int, default=24, help='Длина предсказания')
    parser.add_argument('--journal', default='exp.json', help='Имя файла журнала')
    parser.add_argument('--n_batches', type=int, default=15, help='Количество батчей')
    parser.add_argument('--batch_size', type=int, default=2, help='Размер батча')
    
    args = parser.parse_args()
    
    journal = run_experiment(args.dataset, args.pred_len, args.journal, args.n_batches, args.batch_size)