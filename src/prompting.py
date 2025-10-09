def build_prompt(pred_len: int, batch_size: int, history_md: str) -> str:
    min_seq = 8  # 2^(6-1) для max e_layers=6
    
    return f"""
Подбери {batch_size} РАЗНЫХ конфигураций для pred_len={pred_len}ч.

Датасет: ETTh1 (часовая гранулярность, 7 переменных)

Параметры:
- seq_len: [24, 48, 96, 168, 336, 720]
- label_len: [24, 48, 96, 168, 336, 720]
- e_layers: [2, 3, 4, 6]
- n_heads: [8, 16]
- factor: [3, 5, 8, 10]

ОГРАНИЧЕНИЯ:
1. label_len < seq_len (строго!)
2. seq_len >= {min_seq}
3. n_heads из [8, 16] (делители 512)
4. seq_len >= {pred_len}
5. label_len >= {pred_len // 2}

{history_md}

Верни JSON массив:
{{"lisf_of_configs": [{{"seq_len": <>, "label_len": <>, "e_layers": <>, "n_heads": <>, "factor": <>}}]}}
"""