def build_prompt(pred_len: int, batch_size: int, history_md: str) -> str:
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
2. ЗАПРЕЩЕНО использовать одинаковые конфигурации (в т.ч. из истории)

{history_md}

Для каждой конфигурации сформулируй гипотезу: почему именно эти параметры могут улучшить результат с учетом истории экспериментов.

Верни JSON массив:
{{"list_of_configs": [{{"hypothesis": "<обоснование выбора параметров>", "seq_len": <>, "label_len": <>, "e_layers": <>, "n_heads": <>, "factor": <>}}]}}
"""

# 2. seq_len >= {min_seq}
# 3. n_heads из [8, 16] (делители 512)
# 4. seq_len >= {pred_len}
# 5. label_len >= {pred_len // 2}