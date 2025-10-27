from datetime import datetime
from .optimization_journal import Journal
from .llm_requester import call_llm
from .train import train_informer
from .prompt_builder import build_prompt
from config import GRIDS, build_llm_list_config, get_train_config

def run_experiment(dataset_path, pred_len: int, journal_name: str, grid_name: str = 'GRID12',
                   n_batches: int = 3, batch_size: int = 5, 
                   temperature: float = 1, thinking_budget: int = 2048,
                   ) -> Journal:
    
    print(f"\n{'='*60}\nPRED_LEN={pred_len}ч | {datetime.now()}\n{'='*60}\n")
    
    # Инициализация журнала
    journal = Journal(filename=journal_name)
    start_trial = journal.count_trials()
    total = n_batches * batch_size

    # Получаем сетку параметров и строим датакласс
    params_grid = GRIDS.get(grid_name)
    ListConfigs = build_llm_list_config(params_grid)
    
    for batch in range(n_batches):
        prompt = build_prompt(params_grid, batch_size, journal.get_history_table())
        configs = call_llm(prompt, ListConfigs, temperature=temperature, thinking_budget=thinking_budget)
        
        for i, cfg in enumerate(configs):
            # Конфигурация train состоит из динамических параметров и констант
            train_cfg = get_train_config(cfg, pred_len)

            # Номер текущего эксперимента
            trial_num = start_trial + batch * batch_size + i + 1
            
            # Вывод текущего эксперимента
            #print(f"[{trial_num}/{start_trial + total}] seq={cfg.seq_len} lbl={cfg.label_len} e={cfg.e_layers} h={cfg.n_heads} f={cfg.factor} ", end='')
            params_str = ' '.join(f"{k}={v}" for k, v in cfg.__dict__.items() if k != 'hypothesis')
            print(f"[{trial_num}/{start_trial + total}] {params_str}", end=' ')
            
            # Обучили, занесли в журнал, вывели результат
            measures = train_informer(dataset_path, train_cfg, pred_len)
            journal.add(cfg, measures, trial_num)
            print(f"→ MSE_val={measures['mse_val']:.4f} MAE_val={measures['mae_val']:.4f}\n")
        
        print(f"{'-'*60}")
    
    journal.print_best()
    return journal