import openai
import os
from utils.prompt_templates import build_self_ask_prompt

openai.api_key = os.getenv("OPENAI_API_KEY")

def gpt_self_ask_verifier(claim, evidence):
    prompt = build_self_ask_prompt(claim, evidence)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content'].strip().lower()
