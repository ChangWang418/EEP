import os
from openai import OpenAI
from utils.prompt_templates import build_self_ask_prompt
from dotenv import load_dotenv

# 计算相对路径：从 model/ 跳到 CoT，再跳到 FEVER，再跳到根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))       # ...\CoT\model
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "..", ".."))

# 加载根目录下的 .env
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("❌ OPENROUTER_API_KEY not found. Please check .env file!")

print("✅ Loaded API key prefix:", api_key[:10])  # 调试用

client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)

def gpt_self_ask_verifier(claim, evidence):
    prompt = build_self_ask_prompt(claim, evidence)
    response = client.chat.completions.create(
        model="qwen/qwen3-vl-8b-instruct",
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

