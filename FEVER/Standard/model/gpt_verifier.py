import os
from openai import OpenAI
from utils.prompt_templates import build_self_ask_prompt
from dotenv import load_dotenv

# 计算相对路径：从 model/ 跳到 CoT，再跳到 FEVER，再跳到根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))       # ...\CoT\model
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "..", ".."))

# 加载根目录下的 .env
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("❌ OPENAI_API_KEY not found. Please check .env file!")

print("✅ Loaded API key prefix:", api_key[:10])  # 调试用

client = OpenAI(api_key=api_key)

def gpt_self_ask_verifier(claim, evidence):
    prompt = build_self_ask_prompt(claim, evidence)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",   
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip().lower()
