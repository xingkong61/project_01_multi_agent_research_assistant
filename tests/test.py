"""模型连通性冒烟测试：验证 init_chat_model + .env 配置是否可正常调用。"""
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

# 从项目根目录加载 .env（无论从根目录还是 tests/ 目录运行都生效）
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=True)

model_name = os.getenv("LLM_MODEL", "qwen3.8-flash")
base_url = os.getenv("LLM_BASE_URL")
api_key = os.getenv("LLM_API_KEY")

model = init_chat_model(
    model=model_name,
    model_provider="openai",  # 阿里百炼兼容端点走 OpenAI 协议
    base_url=base_url,
    api_key=api_key,
    temperature=0.4,
)

messages = [HumanMessage(content="只回复两个字：你好")]
res = model.invoke(messages)
print("模型返回：", res.content)
