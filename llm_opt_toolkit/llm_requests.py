import json
from typing import List
from google import genai
from google.genai import types
from config import InformerConfig, GRID, GEMINI_API_KEY, ListConfigs
 
def call_llm(prompt: str, temperature: float = 1, thinking_budget: int = 2048) -> List[InformerConfig]:
    """Запрос к LLM (Gemini/GPT)"""
    client = genai.Client(api_key=GEMINI_API_KEY)
    config = types.GenerateContentConfig(
        system_instruction="You are hyperparameter optimizer. Structrly follow rules in prompt.",
        response_mime_type="application/json",
        response_schema=ListConfigs,
        temperature=temperature,
        thinking_config = types.ThinkingConfig(
            thinking_budget=thinking_budget,
        )
    )
    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=prompt,
        config=config,
    )
    configs_json = json.loads(response.text)
    
    # MOCK: генерируем случайные конфиги
        
    # configs_json = []
    # for _ in range(5):
    #     seq_len = random.choice(GRID['seq_len'])
    #     valid_labels = [l for l in GRID['label_len'] if l < seq_len]
    #     label_len = random.choice(valid_labels) if valid_labels else seq_len // 2
        
    #     configs_json.append({
    #         'seq_len': seq_len,
    #         'label_len': label_len,
    #         'e_layers': random.choice(GRID['e_layers']),
    #         'n_heads': random.choice(GRID['n_heads']),
    #         'factor': random.choice(GRID['factor']),
    #     })
    #print(configs_json)
    #print(type(configs_json))
    return [InformerConfig(**c) for c in configs_json['lisf_of_configs']]