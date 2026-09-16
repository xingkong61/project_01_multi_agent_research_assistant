"""test_citations.py — 引用可信度校验的离线回归测试。

针对模型幻觉编造 arXiv 链接（如未来日期编号 2603.xxxxx）的问题：
- is_implausible_arxiv_id：识别非法/未来 arXiv 编号
- validate_citations：溯源（URL 必须出现在工具返回里）+ 合理性双重过滤
- Researcher 端到端：伪造来源被剔除、可信度下调、错误留痕
全程 mock，不联网。
"""
import os
from datetime import datetime

os.environ.setdefault("LLM_MODEL", "custom/fake")
os.environ.setdefault("LLM_BASE_URL", "http://localhost:1/v1")
os.environ.setdefault("LLM_API_KEY", "fake")

from langchain_core.messages import AIMessage
from langchain_core.tools import tool

import agents

NOW = datetime(2025, 9, 13)


# ------------------------------------------------------------------ #
# arXiv 编号合理性
# ------------------------------------------------------------------ #
def test_implausible_future_and_illegal_ids():
    assert agents.is_implausible_arxiv_id("https://arxiv.org/abs/2603.07612v1", now=NOW)
    assert agents.is_implausible_arxiv_id("https://arxiv.org/abs/2613.03442", now=NOW)   # 月份13
    assert agents.is_implausible_arxiv_id("https://arxiv.org/abs/9999.00001", now=NOW)   # 荒谬年份


def test_plausible_ids_and_non_arxiv():
    assert not agents.is_implausible_arxiv_id("https://arxiv.org/abs/2307.01419", now=NOW)
    assert not agents.is_implausible_arxiv_id("https://arxiv.org/pdf/2501.00001", now=NOW)
    assert not agents.is_implausible_arxiv_id("https://example.com/paper", now=NOW)      # 非 arxiv 不判


# ------------------------------------------------------------------ #
# validate_citations：溯源 + 合理性
# ------------------------------------------------------------------ #
def test_validate_citations_drops_fabricated_and_unprovenanced():
    srcs = [
        {"url": "https://arxiv.org/abs/2307.01419", "title": "Survey"},  # 出现且合理 → 保留
        {"url": "https://arxiv.org/abs/2603.07612", "title": "Fake"},    # 未来编号 → 剔除
        {"url": "https://evil.com/x", "title": "NoProv"},                # 未出现 → 剔除
    ]
    kept, dropped = agents.validate_citations(srcs, {"https://arxiv.org/abs/2307.01419"}, now=NOW)
    assert [s["url"] for s in kept] == ["https://arxiv.org/abs/2307.01419"]
    assert len(dropped) == 2


def test_validate_citations_skips_provenance_when_no_observed():
    # observed_urls 为空/None 时不做溯源（检索全失败的兜底场景），仅查合理性
    srcs = [{"url": "https://arxiv.org/abs/2307.01419"}]
    kept, dropped = agents.validate_citations(srcs, None, now=NOW)
    assert len(kept) == 1 and dropped == []


# ------------------------------------------------------------------ #
# Researcher 端到端：伪造来源被剔除并下调可信度
# ------------------------------------------------------------------ #
@tool
def fake_search(query: str) -> str:
    """搜索"""
    # 真实只返回一个合法 URL；模型若引用别的即为编造
    return "[1] Survey\nURL: https://arxiv.org/abs/2307.01419\n摘要: RAG survey"


class CitingLLM:
    def bind_tools(self, tools, **kw):
        return self

    def invoke(self, messages):
        sysmsg = messages[0].content
        if "资深研究助手" not in sysmsg:
            return AIMessage(content="", tool_calls=[])
        n_search = sum(1 for m in messages if getattr(m, "name", "") == "fake_search")
        if n_search == 0:
            return AIMessage(content="搜", tool_calls=[{
                "name": "fake_search", "args": {"query": "q"}, "id": "s", "type": "tool_call"}])
        # 提交时混入一条真实 + 一条伪造（未来编号）+ 一条无出处
        return AIMessage(content="提交", tool_calls=[{
            "name": "submit_finding",
            "args": {
                "summary": "RAG 综述见 https://arxiv.org/abs/2307.01419，另有 https://arxiv.org/abs/2699.00001。",
                "sources": [
                    {"title": "Survey", "url": "https://arxiv.org/abs/2307.01419"},
                    {"title": "Fake", "url": "https://arxiv.org/abs/2699.00001"},
                    {"title": "Ghost", "url": "https://nope.example/z"},
                ],
                "confidence": "high",
            },
            "id": "f", "type": "tool_call"}])


def test_researcher_filters_bad_citations():
    orig_tools, orig_get = agents.RESEARCH_TOOLS, agents.get_llm
    agents.RESEARCH_TOOLS = [fake_search]
    agents.get_llm = lambda *a, **k: CitingLLM()
    try:
        state = {
            "query": "q",
            "tasks": [{"id": "t1", "description": "A", "rationale": "", "status": "pending"}],
            "findings": {}, "critique": {}, "iteration": 0, "max_iterations": 4,
        }
        upd = agents.node_researcher(state)
    finally:
        agents.RESEARCH_TOOLS, agents.get_llm = orig_tools, orig_get

    f = upd["findings"]["t1"]
    urls = [s["url"] for s in f["sources"]]
    # 只有真实出现过且合理的链接存活
    assert urls == ["https://arxiv.org/abs/2307.01419"], urls
    # 有来源被证伪 → high 下调为 medium
    assert f["confidence"] == "medium", f["confidence"]
    # 剔留在错误留痕中可见
    assert any("不可信引用" in e for e in upd["errors"]), upd["errors"]
