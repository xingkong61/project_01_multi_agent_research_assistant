"""
state.py — 全局共享状态（Blackboard）

所有智能体节点通过读写这同一个状态对象协作：
- Planner 写入 tasks
- Researcher 写入 findings、更新 tasks 状态
- Critic 写入 critique，并把不合格任务重置为 pending
- Writer 写入 report_markdown
- Supervisor 只做调度决策，本身不产生业务数据

列表型字段使用 operator.add 作为 reducer（节点返回新片段时自动追加），
其余字段默认"整体替换"。本模块刻意不依赖 langgraph，便于单测与复用。
"""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict


# ------------------------------------------------------------
# 数据结构
# ------------------------------------------------------------
class Task(TypedDict, total=False):
    """一个研究子任务。"""
    id: str            # 稳定 ID，如 "t1"
    description: str   # 任务描述
    rationale: str     # 为什么要研究它（对最终报告的贡献）
    status: str        # pending / done


class Source(TypedDict, total=False):
    """一条引用来源。"""
    title: str
    url: str
    snippet: str
    published: str     # 可选，论文/文章的发布时间


class Finding(TypedDict, total=False):
    """Researcher 针对某个任务产出的研究结论。"""
    task_id: str
    summary: str                    # 研究结论（Markdown）
    sources: list[Source]           # 引用来源
    confidence: str                 # high / medium / low
    searches: int                   # 调用搜索类工具的次数
    pages_fetched: int              # 抓取网页/论文摘要的次数
    revision: int                   # 第几版（返工后递增）


class CritiqueIssue(TypedDict, total=False):
    """Critic 指出的一条具体问题。task_id 可为空表示全局性问题。"""
    task_id: str
    issue: str


class Critique(TypedDict, total=False):
    """Critic 的结构化评审结论。"""
    decision: str                    # PASS / REVISE / REPLAN
    scores: dict[str, int]           # completeness / accuracy / credibility / depth（1-5）
    issues: list[CritiqueIssue]      # 需要补强的具体问题
    missing_topics: list[str]        # 计划层面缺失、需要新增子任务的主题
    comment: str                     # 总体评语


# ------------------------------------------------------------
# 全局状态
# ------------------------------------------------------------
class ResearchState(TypedDict, total=False):
    # —— 输入 ——
    query: str

    # —— Planner ——
    plan: str
    tasks: list[Task]

    # —— Researcher ——（key = task id）
    findings: dict[str, Finding]

    # —— Critic ——
    critique: Critique
    critique_history: Annotated[list[Critique], operator.add]

    # —— Writer ——
    report_markdown: str

    # —— 流程控制 ——
    iteration: int                   # Critic 已评审轮数
    max_iterations: int              # 最大评审-返工轮数（防死循环）

    # —— 可观测性 ——
    errors: Annotated[list[str], operator.add]
    trace: Annotated[list[str], operator.add]
