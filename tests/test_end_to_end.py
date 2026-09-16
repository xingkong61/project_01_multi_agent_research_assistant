"""
test_end_to_end.py — 用假 LLM + 假工具离线驱动完整 LangGraph 图执行。

覆盖真实调度路径：规划 → 研究 → 评审(REVISE) → 返工 → 复审(PASS) → 成稿。
全程不联网、不调用任何真实模型，因此可安全纳入 CI。
由仓库根目录遗留的临时冒烟脚本 _smoke.py 转正而来。
"""
import os
import re

os.environ.setdefault("LLM_MODEL", "custom/fake")
os.environ.setdefault("LLM_BASE_URL", "http://localhost:1/v1")
os.environ.setdefault("LLM_API_KEY", "fake")

from langchain_core.messages import AIMessage
from langchain_core.tools import tool

import agents
from graph import build_graph


@tool
def fake_search(query: str) -> str:
    """搜索"""
    return f"[1] 论文标题\nURL: https://arxiv.org/abs/123\n摘要: 关于 {query} 的资料"


class FakeLLM:
    def bind_tools(self, tools, **kw):
        return self

    def invoke(self, messages):
        sysmsg = messages[0].content
        human = messages[1].content if len(messages) > 1 else ""

        if "调度员" in sysmsg:
            return AIMessage(content="", tool_calls=[{
                "name": "handoff_to",
                "args": {"next_agent": "researcher", "reason": "继续研究"},
                "id": "h1", "type": "tool_call",
            }])

        if "研究项目的负责人" in sysmsg:
            return AIMessage(content=(
                '{"plan": "测试计划", "tasks": ['
                '{"description": "任务一", "rationale": "r1"},'
                '{"description": "任务二", "rationale": "r2"}]}'
            ))

        if "资深研究助手" in sysmsg:
            tid = re.search(r"ID=(t\d+)", human).group(1)
            step = _calls["researcher"].get(tid, 0)
            _calls["researcher"][tid] = step + 1
            if step % 2 == 0:
                return AIMessage(content="先搜索", tool_calls=[{
                    "name": "fake_search", "args": {"query": tid},
                    "id": f"s-{tid}-{step}", "type": "tool_call"}])
            return AIMessage(content="提交结论", tool_calls=[{
                "name": "submit_finding",
                "args": {
                    "summary": f"{tid} 的研究结论 (来源: https://arxiv.org/abs/123)",
                    "sources": [{"title": "论文", "url": "https://arxiv.org/abs/123"}],
                    "confidence": "high",
                },
                "id": f"f-{tid}-{step}", "type": "tool_call",
            }])

        if "严格的学术评审" in sysmsg:
            _calls["critic"] += 1
            if _calls["critic"] == 1:
                return AIMessage(content=(
                    '{"scores": {"completeness": 3, "accuracy": 4, '
                    '"credibility": 3, "depth": 3},'
                    '"decision": "REVISE",'
                    '"issues": [{"task_id": "t2", "issue": "来源不足，需补强"}],'
                    '"missing_topics": [], "comment": "t2 偏浅"}'
                ))
            return AIMessage(content=(
                '{"scores": {"completeness": 4, "accuracy": 4, '
                '"credibility": 4, "depth": 5},'
                '"decision": "PASS", "issues": [], "missing_topics": [],'
                '"comment": "质量达标"}'
            ))

        return AIMessage(content="# 测试报告\n\n这是最终报告。")


def _fresh_state():
    global _calls
    _calls = {"researcher": {}, "critic": 0}


def test_full_pipeline_offline():
    _fresh_state()
    orig_tools, orig_llm = agents.RESEARCH_TOOLS, agents.get_llm
    agents.RESEARCH_TOOLS = [fake_search]
    agents.get_llm = lambda *a, **k: FakeLLM()
    try:
        g = build_graph()
        state = g.invoke(
            {"query": "测试问题", "max_iterations": 4},
            config={"recursion_limit": 120},
        )
    finally:
        agents.RESEARCH_TOOLS, agents.get_llm = orig_tools, orig_llm

    assert [t["id"] for t in state["tasks"]] == ["t1", "t2"]
    assert all(t["status"] == "done" for t in state["tasks"])
    assert state["iteration"] == 2
    assert state["findings"]["t2"]["revision"] == 2, "t2 应已返工到第 2 版"
    assert state["report_markdown"].startswith("# 测试报告")
    assert state["critique"]["decision"] == "PASS"
    # 每次 fake_search 调用都应计入统计（修复 #2：不再因缓存而漏计）
    assert state["findings"]["t1"]["searches"] >= 1
    n_sources = sum(len(f["sources"]) for f in state["findings"].values())
    assert n_sources >= 2
