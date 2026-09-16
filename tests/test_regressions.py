"""
test_regressions.py — 针对已修复 bug 的离线回归测试。

覆盖：
- #1 Planner 初始规划能真正重置 operator.add 字段 critique_history（Overwrite）
- #3 Critic 评分清洗兼容字符串小数 / 越界夹取 / 非法值跳过
- #4 compute_recursion_limit 随迭代上限动态放大且不低于默认值
全程不联网、不调用真实模型。
"""
import os

os.environ.setdefault("LLM_MODEL", "custom/fake")
os.environ.setdefault("LLM_BASE_URL", "http://localhost:1/v1")
os.environ.setdefault("LLM_API_KEY", "fake")

import json

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Overwrite

import agents
from state import ResearchState


# ------------------------------------------------------------------
# #1 critique_history 重置
# ------------------------------------------------------------------
class _PlannerLLM:
    def bind_tools(self, tools, **kw):
        return self

    def invoke(self, messages):
        payload = {"plan": "p", "tasks": [{"description": "A", "rationale": "r"}]}
        return AIMessage(content=json.dumps(payload), tool_calls=[])


def test_planner_initial_resets_critique_history():
    orig = agents.get_llm
    agents.get_llm = lambda *a, **k: _PlannerLLM()
    try:
        def seed(state):
            return {
                "critique_history": [{"decision": "PASS"}, {"decision": "REVISE"}],
                "trace": ["seed"],
            }

        g = StateGraph(ResearchState)
        g.add_node("seed", seed)
        g.add_node("planner", agents.node_planner)
        g.add_edge(START, "seed")
        g.add_edge("seed", "planner")
        g.add_edge("planner", END)
        out = g.compile().invoke({"query": "q"})
    finally:
        agents.get_llm = orig

    assert out["critique_history"] == [], (
        "初始规划应通过 Overwrite 清空评审历史，而非残留旧值"
    )
    assert out["findings"] == {}


def test_planner_returns_overwrite_sentinel():
    # 直接检查 node_planner 初始分支返回值携带 Overwrite
    orig = agents.get_llm
    agents.get_llm = lambda *a, **k: _PlannerLLM()
    try:
        upd = agents.node_planner({"query": "q", "tasks": []})
    finally:
        agents.get_llm = orig
    assert isinstance(upd["critique_history"], Overwrite)


def test_merge_update_handles_overwrite():
    # main.run() 的 stream 合并必须识别 Overwrite：整体替换而非 .extend()
    from main import _merge_update

    latest = {"critique_history": [{"decision": "PASS"}, {"decision": "REVISE"}],
              "trace": ["a"]}
    _merge_update(latest, {"critique_history": Overwrite([]), "trace": ["b"]})
    assert latest["critique_history"] == [], "Overwrite 应清空，且不得抛 AttributeError"
    assert latest["trace"] == ["a", "b"], "普通 add-reducer 字段仍按追加合并"


# ------------------------------------------------------------------
# #3 Critic 评分清洗
# ------------------------------------------------------------------
class _CriticLLM:
    def __init__(self, payload):
        self.payload = payload

    def bind_tools(self, tools, **kw):
        return self

    def invoke(self, messages):
        return AIMessage(content=json.dumps(self.payload), tool_calls=[])


def _run_critic(scores):
    orig = agents.get_llm
    agents.get_llm = lambda *a, **k: _CriticLLM({
        "scores": scores, "decision": "PASS",
        "issues": [], "missing_topics": [], "comment": "ok",
    })
    try:
        state = {
            "query": "q",
            "tasks": [{"id": "t1", "description": "A", "status": "done"}],
            "findings": {"t1": {"task_id": "t1", "summary": "s", "sources": [
                {"url": "https://a.com"}], "confidence": "high", "revision": 1}},
            "iteration": 0, "max_iterations": 4,
        }
        return agents.node_critic(state)["critique"]["scores"]
    finally:
        agents.get_llm = orig


def test_critic_scores_accept_string_decimals():
    cleaned = _run_critic({
        "completeness": "5", "accuracy": 9,
        "credibility": None, "depth": "3.5",
    })
    # "3.5" 不再被丢弃；9 夹到 5；None 跳过
    assert cleaned["depth"] == 4          # round(3.5) -> 4
    assert cleaned["accuracy"] == 5       # clamp
    assert cleaned["completeness"] == 5
    assert "credibility" not in cleaned   # None 被剔除


def test_critic_scores_ignore_bool_and_text():
    cleaned = _run_critic({
        "completeness": True, "accuracy": "high", "depth": 2,
    })
    assert cleaned == {"depth": 2}


# ------------------------------------------------------------------
# #4 recursion_limit 动态放大
# ------------------------------------------------------------------
def test_compute_recursion_limit_scales_with_iterations():
    from graph import DEFAULT_RECURSION_LIMIT, compute_recursion_limit

    lo = compute_recursion_limit(1)
    hi = compute_recursion_limit(8)
    assert hi > lo, "更高迭代上限应得到更大步数预算"
    assert lo >= DEFAULT_RECURSION_LIMIT, "不得低于保守默认值"


# ------------------------------------------------------------------ #
# describe_exc：把 openai/网络异常翻译成带状态码与中文提示的可读信息
# ------------------------------------------------------------------ #
class _FakeResp:
    def __init__(self, status, text):
        self.status_code = status
        self.text = text


def test_describe_exc_surfaces_status_and_hint():
    from agents import describe_exc

    class RateLimit(Exception):
        status_code = 429
        response = _FakeResp(429, '{"error": {"message": "You exceeded your current quota"}}')

    desc = describe_exc(RateLimit())
    assert "429" in desc
    assert "quota" in desc.lower()
    assert "限流" in desc or "额度" in desc, "应给出中文可操作提示"


def test_describe_exc_plain_exception_still_readable():
    from agents import describe_exc

    desc = describe_exc(ValueError("boom"))
    assert "ValueError" in desc and "boom" in desc
