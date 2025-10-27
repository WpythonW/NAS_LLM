import typing as T
import json

from google import genai
from google.genai import types
from mistralai import Mistral

from config import InformerConfig, GRID, GEMINI_API_KEY, MISTRAL_API_KEY, ListConfigs

class BaseLLM:
    def call_llm(
        self
    ) -> T.List[InformerConfig]:
        raise NotImplementedError

    def build_prompt(self, pred_len: int, batch_size: int, history_md: str) -> str:
        return f"""
    Подбери {batch_size} РАЗНЫХ конфигураций для pred_len={pred_len}ч.

    Датасет: ETTh (часовая гранулярность)

    Параметры:
    - seq_len: [24, 48, 96, 168, 336, 720]
    - label_len: [24, 48, 96, 168, 336, 720]
    - e_layers: [2, 3, 4, 6]
    - n_heads: [8, 16]
    - factor: [3, 5, 8, 10]

    Определение параметров:
    | Parameter name | Description of parameter |
    |----------------|--------------------------|
    | seq_len | Input sequence length of Informer encoder|
    | label_len | Start token length of Informer decoder|
    | e_layers | Num of encoder layers|
    | n_heads | Num of heads|
    | factor | The factor parameter controls how many query points are selected for attention computation: u = factor * log(L) top queries are chosen. Smaller values (factor=3) mean faster computation but potential loss of dependencies, while larger values (factor=8-10) capture more dependencies but with diminishing returns|


    ОГРАНИЧЕНИЯ:
    1. label_len < seq_len (строго!)
    2. ЗАПРЕЩЕНО использовать одинаковые конфигурации (в т.ч. из истории)

    {history_md}

    Для каждой конфигурации сформулируй краткую гипотезу(1-2 предложения): почему именно эти параметры могут улучшить результат с учетом истории экспериментов.

    Верни JSON массив:
    {{"list_of_configs": [{{"hypothesis": "<обоснование выбора параметров>", "seq_len": <>, "label_len": <>, "e_layers": <>, "n_heads": <>, "factor": <>}}]}}
    """


class GeminiAgent(BaseLLM):
    def __init__(self):
        self.client = genai.Client(api_key=GEMINI_API_KEY)
    
    def call_llm(self, prompt: str, temperature: float = 1, thinking_budget: int = 2048) -> T.List[InformerConfig]:
        """Запрос к LLM (Gemini/GPT)"""
        config = types.GenerateContentConfig(
            system_instruction="You are hyperparameter optimizer. Structrly follow rules in prompt.",
            response_mime_type="application/json",
            response_schema=ListConfigs,
            temperature=temperature,
            thinking_config = types.ThinkingConfig(
                thinking_budget=thinking_budget,
            )
        )
        response = self.client.models.generate_content(
            model="gemini-flash-latest",
            contents=prompt,
            config=config,
        )
        configs_json = json.loads(response.text)
        return [InformerConfig(**c) for c in configs_json['lisf_of_configs']]
    

class MistralAgent(BaseLLM):
    def __init__(self):
        self.client = Mistral(api_key=MISTRAL_API_KEY)
    
    def call_llm(self, prompt: str, temperature: float = 1, thinking_budget: int = 2048) -> T.List[InformerConfig]:
        """Запрос к LLM (Gemini/GPT)"""

        response = self.client.chat.parse(
            model = "mistral-large-2411",
            messages = [
                {
                    "role": "system",
                    "content": "You are hyperparameter optimizer. Structure follow rules in prompt.",
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            response_format=ListConfigs,
            temperature=temperature,
        )
        # print(response.choices[0].message.parsed.lisf_of_configs)
       
        return response.choices[0].message.parsed.lisf_of_configs