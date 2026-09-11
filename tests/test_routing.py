"""
tests/test_routing.py — 核心路由逻辑的单元测试
测试 Critic 评审 → 路由决策的边界情况。
"""
import pytest
from nodes import route_after_critic
from state import ResearchState


class TestCriticRouter:
    """
    测试条件边路由函数 route_after_critic 的行为。
    
    路由规则：
    - revision_needed == True  → 返回 "researcher"（打回重新研究）
    - revision_needed == False → 返回 "writer"（进入写作）
    """

    # ---- 通过场景 ----

    def test_revision_not_needed_returns_writer(self):
        """当研究质量合格时，应进入 writer"""
        state: ResearchState = {
            "query": "RAG 技术的最新进展",
            "tasks": ["RAG 核心原理", "RAG 最新优化方法"],
            "completed_tasks": {
                "RAG 核心原理": {"content": "RAG 是检索增强生成...", "sources": ["https://example.com"]},
                "RAG 最新优化方法": {"content": "CRAG 是自适应检索...", "sources": ["https://example.com"]},
            },
            "revision_needed": False,
            "loop_count": 1,
            "critiques": ["研究质量良好"],
            "final_report": "",
            "errors": [],
        }
        result = route_after_critic(state)
        assert result == "writer"

    def test_revision_not_needed_with_empty_critiques(self):
        """即使没有评审记录，质量不达标时 revision_needed=False → writer"""
        state: ResearchState = {
            "revision_needed": False,
        }
        result = route_after_critic(state)
        assert result == "writer"

    # ---- 不合格 / 返工场景 ----

    def test_revision_needed_returns_researcher(self):
        """当研究质量不合格时，应打回 researcher"""
        state: ResearchState = {
            "revision_needed": True,
            "critiques": ["缺少 2024 年最新数据，需要补充"],
            "loop_count": 1,
        }
        result = route_after_critic(state)
        assert result == "researcher"

    def test_revision_needed_missing_task(self):
        """当有任务未完成时，应打回 researcher"""
        state: ResearchState = {
            "tasks": ["任务A", "任务B", "任务C"],
            "completed_tasks": {
                "任务A": {"content": "已完成A", "sources": []},
            },
            "revision_needed": True,
            "critiques": ["任务B 和任务C 尚未完成"],
        }
        result = route_after_critic(state)
        assert result == "researcher"

    def test_revision_needed_with_full_completion(self):
        """即使所有任务完成，如果 revision_needed=True 仍应返工"""
        state: ResearchState = {
            "tasks": ["任务A"],
            "completed_tasks": {"任务A": {"content": "完成", "sources": []}},
            "revision_needed": True,
            "critiques": ["内容太浅，需要更深入分析"],
        }
        result = route_after_critic(state)
        assert result == "researcher"

    # ---- 边界场景 ----

    def test_revision_needed_key_missing_defaults_to_writer(self):
        """revision_needed 字段缺失时，默认进入 writer（安全默认值）"""
        state: ResearchState = {
            "query": "测试",
            "revision_needed": None,
        }
        # route_after_critic 使用 state.get("revision_needed", False)
        # None 被判定为 False → writer
        result = route_after_critic(state)
        assert result == "writer"

    def test_revision_needed_explicit_false(self):
        """revision_needed 显式为 False 时，进入 writer"""
        state: ResearchState = {"revision_needed": False}
        result = route_after_critic(state)
        assert result == "writer"


class TestStateSchema:
    """测试状态 schema 的完整性"""

    def test_complete_state_has_all_required_fields(self):
        """完整的 ResearchState 包含所有必要字段"""
        state: ResearchState = {
            "query": "测试问题",
            "tasks": ["任务1", "任务2"],
            "plan": "测试计划",
            "completed_tasks": {"任务1": {"content": "...", "sources": []}},
            "critiques": [],
            "revision_needed": False,
            "loop_count": 0,
            "final_report": "",
            "errors": [],
        }
        assert "query" in state
        assert "tasks" in state
        assert "revision_needed" in state
        assert "loop_count" in state

    def test_state_accepts_empty_completed_tasks(self):
        """空 completed_tasks 是合法的初始状态"""
        state: ResearchState = {
            "query": "测试",
            "completed_tasks": {},
        }
        assert isinstance(state["completed_tasks"], dict)
        assert len(state["completed_tasks"]) == 0
