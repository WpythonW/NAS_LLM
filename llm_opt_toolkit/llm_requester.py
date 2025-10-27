import json
from typing import List
from google import genai
from google.genai import types
from config import GEMINI_API_KEY
from pydantic import BaseModel

def call_llm(prompt: str, ListConfigs: BaseModel, temperature: float = 1, thinking_budget: int = 2048):
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    config = types.GenerateContentConfig(
        system_instruction="You are hyperparameter optimizer. Strictly follow rules in prompt.",
        response_mime_type="application/json",
        response_schema=ListConfigs,
        temperature=temperature,
        thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget)
    )

    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=prompt,
        config=config
    )
    
    return response.parsed.list_of_configs