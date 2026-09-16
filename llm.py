"""
llm.py — LLM 工厂（基于 langchain.chat_models.init_chat_model）

通过 "供应商/模型名" 字符串选择模型，统一走 OpenAI 兼容协议：

    qwen3.8-flash                   → 默认模型，走 LLM_BASE_URL / LLM_API_KEY（兼容端点）
    groq/llama-3.3-70b-versatile   → Groq
    openai/gpt-4o                  → OpenAI
    deepseek/deepseek-chat         → DeepSeek
    custom/<model>                 → 任意 OpenAI 兼容端点（LLM_BASE_URL / LLM_API_KEY）

也可以直接用环境变量 LLM_MODEL 指定，例如：
    LLM_MODEL=qwen3.8-flash（更快更省）或 qwen3.8-max（深度推理）
未写 "供应商/" 前缀时，若设置了 LLM_BASE_URL，则自动按自定义兼容端点处理。

Groq / DeepSeek 同样是 OpenAI 兼容接口，因此统一使用 model_provider="openai"
并显式传入 base_url，无需安装 langchain-groq / langchain-deepseek。
"""
from __future__ import annotations

import contextvars
import os

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

# 每次请求的模型覆盖：用 contextvar 承载，天然并发隔离，
# 服务层（Web/API）可 per-request 设置而无需改写进程级环境变量。
_request_model: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "llm_request_model", default=None
)


def set_request_model(model: str | None) -> contextvars.Token:
    """设置当前上下文使用的模型标识，返回 token 以便 reset 复原。"""
    return _request_model.set(model)

# provider -> (base_url, API Key 环境变量名)
_PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "openai": (None, "OPENAI_API_KEY"),
    "deepseek": ("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
    # 国内常见兼容端点示例：
    # "moonshot": ("https://api.moonshot.cn/v1", "MOONSHOT_API_KEY"),
}

DEFAULT_MODEL = "qwen3.8-flash"


def get_llm(model: str | None = None, temperature: float = 0.4,
            max_tokens: int = 4096, thinking: bool = True) -> BaseChatModel:
    """
    通过 init_chat_model 创建一个聊天模型实例（统一为 OpenAI 兼容协议）。

    :param model: "provider/model-name"；优先级 显式参数 > 请求上下文(set_request_model)
                  > LLM_MODEL 环境变量 > 默认值
    :param temperature: 采样温度
    :param max_tokens: 单次输出上限
    :param thinking: 是否开启深度思考（仅对支持的模型生效，如 qwen3 系列）。
                     思考会大量消耗 token 并显著增加耗时；材料加工类节点
                     （Writer/Critic/Supervisor）建议关闭。
    """
    spec = (model or _request_model.get() or os.getenv("LLM_MODEL") or DEFAULT_MODEL).strip()

    if "/" in spec:
        provider, model_name = spec.split("/", 1)
    else:
        # 没写前缀时：配置了 LLM_BASE_URL 就走自定义兼容端点，否则按官方 OpenAI 处理
        model_name = spec
        provider = "custom" if os.getenv("LLM_BASE_URL") else "openai"

    if provider == "custom":
        base_url = os.getenv("LLM_BASE_URL", "")
        api_key = os.getenv("LLM_API_KEY", "")
        if not base_url:
            raise ValueError("使用 custom 供应商时必须设置 LLM_BASE_URL")
    else:
        if provider not in _PROVIDERS:
            supported = ", ".join(sorted(_PROVIDERS) + ["custom"])
            raise ValueError(f"不支持的供应商：{provider}（支持：{supported}）")
        base_url, key_env = _PROVIDERS[provider]
        api_key = os.getenv(key_env, "")
        if not api_key:
            raise RuntimeError(
                f"未设置环境变量 {key_env}，请在 .env 中配置（参考 .env.example）"
            )

    # 推理类模型（如 qwen3.8-max）单次生成常需 30~120s，非流式请求会在整块响应
    # 到达前干等并触发读超时。streaming=True 让底层走 SSE：数据块持续到达，读超时
    # 按"块间隔"计算而非总时长，.invoke() 仍返回聚合后的完整 AIMessage，业务无感。
    timeout = float(os.getenv("LLM_TIMEOUT", "180"))
    # 推理模型的"思考"同样计入 max_tokens 预算：预算太小会被思考耗尽，
    # 导致正文为空或工具调用被截断。默认 8192，可通过 LLM_MAX_TOKENS 调整。
    max_tokens = int(os.getenv("LLM_MAX_TOKENS", str(max_tokens if max_tokens != 4096 else 8192)))

    # qwen3 系列等支持 enable_thinking 参数；思考开启时 token 消耗和耗时会成倍增长
    kwargs: dict = {
        "model": model_name,
        # Groq / DeepSeek / 自定义端点均为 OpenAI 兼容协议，统一用 openai provider
        "model_provider": "openai",
        "api_key": api_key or "not-configured",
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": timeout,
        "max_retries": 2,
        # 推理模型单次生成常需 30~120s：streaming 走 SSE，读超时按"块间隔"计算，
        # .invoke() 仍返回聚合后的完整 AIMessage，业务代码无感
        "streaming": True,
        "stream_usage": True,
    }
    if base_url:
        kwargs["base_url"] = base_url
    if not thinking:
        kwargs["extra_body"] = {"enable_thinking": False}

    return init_chat_model(**kwargs)
