"""test_truncation.py — 结论截断检测与自动放大预算重试的离线回归测试。

复现并锁定根因：推理模型思考 token 挤占正文预算 → submit_finding 的 summary
被 max_tokens 物理砍断 → Critic 反复打回同一任务、评审空转到耗尽轮数。
修复后 Researcher 应检测到截断并以更大 max_tokens 重跑提交，而非接受残缺结论。
全程 mock LLM，不联网。
"""
import os

os.environ.setdefault("LLM_MODEL", "custom/fake")
os.environ.setdefault("LLM_BASE_URL", "http://localhost:1/v1")
os.environ.setdefault("LLM_API_KEY", "fake")

from langchain_core.messages import AIMessage
from langchain_core.tools import tool

import agents


# ------------------------------------------------------------------ #
# 纯启发式：_looks_truncated / _finish_reason
# ------------------------------------------------------------------ #
def test_looks_truncated_flags_mid_sentence():
    cut = "Gao et al. 归纳出 RAG 的三种基本范式包括 Naive RAG、Advance RAG 以及 Modular"
    assert agents._looks_truncated(cut) is True


def test_looks_truncated_accepts_normal_endings():
    ok = "综上所述，RAG 的核心范式可归纳为三类。"
    assert agents._looks_truncated(ok) is False
    # 带 markdown 代码块闭合 / 列表项结尾也不误伤
    assert agents._looks_truncated("结论要点如下：\n- 检索增强\n- 生成质量控制。") is False
    # 过短文本一律不判截断（避免误伤简短但完整的结论）
    assert agents._looks_truncated("资料不足。") is False


def test_finish_reason_reads_length():
    m = AIMessage(content="x")
    m.response_metadata = {"finish_reason": "length"}
    assert agents._finish_reason(m) == "length"
    m2 = AIMessage(content="x")
    m2.response_metadata = {"finish_reason": "stop"}
    assert agents._finish_reason(m2) == "stop"


# ------------------------------------------------------------------ #
# ReAct 循环：首次提交被 length 截断 → 放大预算重跑 → 第二次完整提交被接受
# ------------------------------------------------------------------ #
@tool
def fake_search(query: str) -> str:
    """搜索"""
    return f"[1] URL: https://arxiv.org/abs/{query}"


class TruncatingLLM:
    """脚本化：搜索 → 截断提交(finish_reason=length) → 完整提交。"""
    def __init__(self):
        self.calls = 0

    def bind_tools(self, tools, **kw):
        return self

    def invoke(self, messages):
        sysmsg = messages[0].content
        if "资深研究助手" not in sysmsg:
            return AIMessage(content="", tool_calls=[])
        self.calls += 1
        if self.calls == 1:
            return AIMessage(content="先搜索", tool_calls=[{
                "name": "fake_search", "args": {"query": "q"}, "id": "s1", "type": "tool_call"}])
        if self.calls == 2:
            m = AIMessage(content="", tool_calls=[{
                "name": "submit_finding",
                "args": {"summary": "RAG 的三种基本范式是 Naive RAG、Advanced RAG 与 Modular RAG，其中前两者强调检索",
                         "sources": [], "confidence": "high"},
                "id": "f1", "type": "tool_call"}])
            m.response_metadata = {"finish_reason": "length"}
            return m
        return AIMessage(content="", tool_calls=[{
            "name": "submit_finding",
            "args": {"summary": "RAG 的三种基本范式是 Naive、Advanced 与 Modular RAG，各自针对检索质量、生成忠实度与系统模块化做了改进。",
                     "sources": [{"title": "Survey", "url": "https://arxiv.org/abs/2307.01419"}],
                     "confidence": "high"},
            "id": "f2", "type": "tool_call"}])


def test_researcher_retries_on_truncated_submission(monkeypatch):
    orig_tools = agents.RESEARCH_TOOLS
    agents.RESEARCH_TOOLS = [fake_search]
    spy = TruncatingLLM()
    seen_tokens = []

    def fake_get_llm(model=None, temperature=0.4, max_tokens=4096, thinking=True):
        seen_tokens.append(max_tokens)
        return spy

    orig_get = agents.get_llm
    agents.get_llm = fake_get_llm
    try:
        state = {
            "query": "q",
            "tasks": [{"id": "t1", "description": "A", "rationale": "", "status": "pending"}],
            "findings": {}, "critique": {}, "iteration": 0, "max_iterations": 4,
        }
        upd = agents.node_researcher(state)
    finally:
        agents.RESEARCH_TOOLS = orig_tools
        agents.get_llm = orig_get

    f = upd["findings"]["t1"]
    # 最终提交的应是完整版（以句号收尾），不是半句
    assert f["summary"].rstrip().endswith("。"), f"应接受完整结论，实际：{f['summary'][-20:]}"
    assert "Modular RAG" in f["summary"]
    # 记录到一次截断重试
    assert any("被截断" in e for e in upd["errors"]), upd["errors"]
    # 第二次 get_llm 的 max_tokens 应是第一次的两倍（预算放大）
    assert len(seen_tokens) >= 2 and seen_tokens[1] == seen_tokens[0] * 2, seen_tokens
