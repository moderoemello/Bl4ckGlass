import os
from dotenv import load_dotenv
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def ask_llm(prompt, model="gpt-4"):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        return f"[LLM Error] {e}"
