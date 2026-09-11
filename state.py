"""
state.py — 共享状态定义
所有节点的读写都围绕这个 schema 进行，是 LangGraph 的核心数据结构。
"""
from typing import TypedDict, Annotated, Sequence
from langgraph.graph import add_messages


class ResearchState(TypedDict, total=False):
    """
    多智能体研究助手的状态 schema
    每个字段由不同的 Agent 读写，形成流水线式的状态传递。
    """

    # 原始用户问题
    query: str

    # Planner 产出的子任务列表
    tasks: Annotated[list[str], add_messages]
    # Planner 分解后的任务计划说明
    plan: str

    # 已完成的任务结果：{ "task_name": { "content": "...", "sources": [...] } }
    completed_tasks: dict

    # Critic 对已完成任务的评审意见
    critiques: Annotated[list[str], add_messages]

    # 当前状态下的评审结论：是否通过
    revision_needed: bool

    # 最大循环次数（防止死循环）
    loop_count: int

    # Writer 最终生成的结构化报告
    final_report: str

    # 执行过程中的错误信息
    errors: list[str]
