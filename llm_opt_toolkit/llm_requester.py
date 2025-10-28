from google import genai
from google.genai import types
from config import GEMINI_API_KEY
from pydantic import BaseModel

from google import genai
from google.genai import types
from google.genai.errors import ServerError
from pydantic import BaseModel
import time

def call_llm(prompt: str, ListConfigs: BaseModel, temperature: float = 1, thinking_budget: int = 2048, max_retries: int = 100):
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    config = types.GenerateContentConfig(
        system_instruction="You are hyperparameter optimizer. Strictly follow rules in prompt.",
        response_mime_type="application/json",
        response_schema=ListConfigs,
        temperature=temperature,
        thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget)
    )

    for attempt in range(max_retries):
        try:
            print('Attempting to call LLM, try', attempt + 1)
            response = client.models.generate_content(
                model="gemini-flash-latest",
                contents=prompt,
                config=config
            )
            return response.parsed.list_of_configs
        
        except ServerError as e: time.sleep(2)
            #raise Exception('WTF model didn-t respond??')