import argparse
from llm_opt_toolkit.optimizator import run_experiment

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='Путь к датасету')
    parser.add_argument('--pred_len', type=int, default=24, help='Длина предсказания')
    parser.add_argument('--journal', default='exp.json', help='Имя файла журнала')
    parser.add_argument('--n_batches', type=int, default=15, help='Количество батчей')
    parser.add_argument('--batch_size', type=int, default=2, help='Размер батча')
    parser.add_argument('--temperature', type=float, default=1, help='Температура для LLM')
    parser.add_argument('--thinking_budget', type=int, default=2048, help='Thinking budget для LLM')
    parser.add_argument('--expr_num', type=int, default=2048, help='2 или 3 эксперимент(в 3 архитектурные параметры)')

    
    args = parser.parse_args()
    
    journal = run_experiment(args.dataset, args.pred_len, args.journal, args.n_batches, 
                             args.batch_size , args.temperature, args.thinking_budget,
                             args.expr_num)