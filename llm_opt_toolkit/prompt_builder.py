import pandas as pd
from config import PARAM_DESCRIPTIONS

def _format_params_table(params_grid: dict) -> str:
    df = pd.DataFrame(list(params_grid.items()), columns=['Parameter', 'Values'])
    df['Values'] = df['Values'].astype(str)
    return df.to_markdown(index=False)

def _format_descriptions_table(params_grid: dict) -> str:
    data = {k: PARAM_DESCRIPTIONS[k] for k in params_grid.keys()}
    df = pd.DataFrame(list(data.items()), columns=['Parameter', 'Description'])
    return df.to_markdown(index=False)

def _format_json_example(params_grid: dict) -> str:
    fields = ', '.join(f'"{k}": <>' for k in params_grid.keys())
    return f'{{"hypothesis": "<обоснование>", {fields}}}'

def build_prompt(params_grid: dict, batch_size: int, history_md: str) -> str:
    return f"""Подбери {batch_size} РАЗНЫХ конфигураций.

Датасет: ETTh (часовая гранулярность)

Параметры:
{_format_params_table(params_grid)}

Определение параметров:
{_format_descriptions_table(params_grid)}

ОГРАНИЧЕНИЯ:
1. label_len < seq_len (строго!)
2. ЗАПРЕЩЕНЫ дубликаты (в т.ч. из истории)

{history_md}

Для каждой конфигурации дай краткую гипотезу (1-2 предложения).

JSON: {{"list_of_configs": [{_format_json_example(params_grid)}]}}"""