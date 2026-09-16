"""test_stream_core.py — 验证 main.stream_research 流式核心（离线，mock LLM）。

覆盖：逐步产出 trace、最终 done 携带报告、以及 model 经 contextvar 注入后
get_llm 能读到（并发隔离的关键改造）。
"""
import os
import re

os.environ.setdefault("LLM_MODEL", "custom/fake")
os.environ.setdefault("LLM_BASE_URL", "http://localhost:1/v1")
os.environ.setdefault("LLM_API_KEY", "fake")

from langchain_core.messages import AIMessage
from langchain_core.tools import tool

import agents
import llm
import main as m


@tool
def fake_search(query: str) -> str:
    """搜索"""
    return f"[1] 标题\nURL: https://arxiv.org/abs/9\n摘要: {query}"


class FakeLLM:
    def __init__(self):
        self._rc = {"n": 0}

    def bind_tools(self, tools, **kw):
        return self

    def invoke(self, messages):
        sysmsg = messages[0].content
        human = messages[1].content if len(messages) > 1 else ""
        if "调度员" in sysmsg:
            return AIMessage(content="", tool_calls=[{"name": "handoff_to",
                "args": {"next_agent": "researcher", "reason": "继续"}, "id": "h", "type": "tool_call"}])
        if "研究项目的负责人" in sysmsg:
            return AIMessage(content='{"plan":"p","tasks":[{"description":"A","rationale":"r"}]}')
        if "资深研究助手" in sysmsg:
            step = self._rc["n"]; self._rc["n"] = step + 1
            if step % 2 == 0:
                return AIMessage(content="搜", tool_calls=[{"name": "fake_search",
                    "args": {"query": "q"}, "id": "s", "type": "tool_call"}])
            return AIMessage(content="提交", tool_calls=[{"name": "submit_finding",
                "args": {"summary": "结论 https://arxiv.org/abs/9", "sources": [], "confidence": "high"},
                "id": "f", "type": "tool_call"}])
        if "严格的学术评审" in sysmsg:
            return AIMessage(content='{"scores":{"completeness":4,"accuracy":4,'
                '"credibility":4,"depth":4},"decision":"PASS","issues":[],'
                '"missing_topics":[],"comment":"ok"}')
        return AIMessage(content="# 报告\n最终。")


def test_stream_research_yields_trace_and_done():
    orig_tools, orig_llm = agents.RESEARCH_TOOLS, agents.get_llm
    agents.RESEARCH_TOOLS = [fake_search]
    agents.get_llm = lambda *a, **k: FakeLLM()
    try:
        events = list(m.stream_research("测试问题", None, 4))
    finally:
        agents.RESEARCH_TOOLS, agents.get_llm = orig_tools, orig_llm

    kinds = [k for k, _ in events]
    assert "trace" in kinds and "done" in kinds
    # done 是最后一个事件，携带最终状态与报告
    last_kind, last_payload = events[-1]
    assert last_kind == "done"
    assert last_payload["report_markdown"].startswith("# 报告")
    # status 快照应反映任务完成进度
    statuses = [p for k, p in events if k == "status"]
    assert any(s["tasks_total"] >= 1 for s in statuses)


def test_model_injected_via_contextvar():
    # stream_research 传 model 时，图内 get_llm 应读到该 model（而非 env）
    seen = {}

    class SpyLLM(FakeLLM):
        pass

    def fake_get_llm(model=None, temperature=0.4, max_tokens=4096, thinking=True):
        # 复刻真实解析优先级：显式参数 > contextvar > env
        spec = (model or llm._request_model.get() or os.getenv("LLM_MODEL") or llm.DEFAULT_MODEL)
        seen["spec"] = spec
        return SpyLLM()

    orig_tools, orig_llm = agents.RESEARCH_TOOLS, agents.get_llm
    agents.RESEARCH_TOOLS = [fake_search]
    agents.get_llm = fake_get_llm
    try:
        list(m.stream_research("q", "groq/llama-3.3-70b-versatile", 2))
    finally:
        agents.RESEARCH_TOOLS, agents.get_llm = orig_tools, orig_llm

    assert seen.get("spec") == "groq/llama-3.3-70b-versatile", \
        "请求级 model 应通过 contextvar 传入并被 get_llm 读取"
