from datetime import datetime
import numpy as np
import pandas as pd
import torch
from .optimization_journal import Journal
from .train import train_informer
from .llm import GeminiAgent, MistralAgent

def run_experiment(dataset_path, pred_len: int, journal_name: str, 
                   n_batches: int = 3, batch_size: int = 5, 
                   temperature: float = 1, thinking_budget: int = 2048, provider: str = "gemini") -> Journal:
    
    print(f"\n{'='*60}\nPRED_LEN={pred_len}ч | {datetime.now()}\n{'='*60}\n")
    
    journal = Journal(filename=journal_name)
    start_trial = journal.count_trials()
    total = n_batches * batch_size
    trials = np.arange(start_trial + 1, start_trial + total + 1)
    
    provider_map = {
        "gemini": GeminiAgent,
        "mistral": MistralAgent
    }
    llm_agent = provider_map.get(provider, None)
    # print(llm_agent)

    if not llm_agent:
        raise Exception("provider должен быть 'gemini' или 'mistral'")

    llm_agent = llm_agent()

    for idx, trial_num in enumerate(trials):
        
        if idx % batch_size == 0:
            prompt = llm_agent.build_prompt(pred_len, batch_size, journal.get_history_table())
            #print(prompt)
            configs = llm_agent.call_llm(prompt, temperature=temperature, thinking_budget=thinking_budget)
        
        cfg = configs[idx % batch_size]
        print(f"[{trial_num}/{start_trial + total}] seq={cfg.seq_len} lbl={cfg.label_len} e={cfg.e_layers} h={cfg.n_heads} f={cfg.factor} ", end='')
        
        measures = train_informer(dataset_path, cfg, pred_len)
        journal.add(cfg, measures, trial_num)
        
        print(f"→ MSE_val={measures['mse_val']:.4f} MAE_val={measures['mae_val']:.4f}\n")
        
        if (idx + 1) % batch_size == 0:
            print(f"{'-'*60}")
        torch.cuda.empty_cache()
    journal.print_best()
    return journal