"""
graph.py — LangGraph 工作流（Supervisor 拓扑）

拓扑不是写死的线性流水线，而是一个以 Supervisor 为中心的星型结构：
Supervisor 本身是 LLM 节点，在运行时基于黑板状态动态决定 handoff 给谁；
所有 worker 完成后都回到 Supervisor，直到它决定交给 Writer 并结束。

  START → supervisor ◁────────────────────────────┐
              │ LLM 运行时选路（规则护栏兜底）       │
   ┌──────────┼──────────────┬──────────────┐     │
   ▼          ▼              ▼              ▼     │
 planner  researcher      critic         writer ─┴→ END
   └──────────┴──────────────┴──── worker 完成后回 supervisor

图本身只声明“谁可能 handoff 给谁”（拓扑可能性），
“这一步实际去哪”由 Supervisor 智能体在运行时决定（Command(goto=...)）。
"""
import os
import sys

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

# 直接 python graph.py 时也能导入同目录模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents import (
    node_critic,
    node_planner,
    node_researcher,
    node_supervisor,
    node_writer,
)
from state import ResearchState

load_dotenv()

# 图步数上限：N 个任务 + 多轮评审返工时节点跳转次数可能很多，
# main.py 会按任务量/迭代上限动态传入更大的值（见 compute_recursion_limit），
# 这里只给一个保守的默认值。
DEFAULT_RECURSION_LIMIT = 120


def compute_recursion_limit(max_iterations: int, expected_tasks: int = 6) -> int:
    """按最坏路径估算图步数预算，避免任务多/返工轮数多时触顶抛 GraphRecursionError。

    每轮"评审→返工"最多产生：supervisor + researcher(每个待办任务一次)
    + supervisor + critic ≈ 2 + 2·tasks 步；再叠加初始规划、补题(planner)、
    最终 writer 与若干 supervisor 单跳，留足余量并设下限。
    """
    per_round = 2 + 2 * max(1, expected_tasks)
    total = (max_iterations + 1) * per_round + 2 * expected_tasks + 10
    return max(DEFAULT_RECURSION_LIMIT, total)


def build_graph():
    """构建并编译研究智能体工作流图。"""
    g = StateGraph(ResearchState)

    g.add_node("supervisor", node_supervisor)
    g.add_node("planner", node_planner)
    g.add_node("researcher", node_researcher)
    g.add_node("critic", node_critic)
    g.add_node("writer", node_writer)

    # 入口：先问调度员
    g.add_edge(START, "supervisor")
    # 每个 worker 干完活都回到调度员汇报
    g.add_edge("planner", "supervisor")
    g.add_edge("researcher", "supervisor")
    g.add_edge("critic", "supervisor")
    # 报告写完即结束
    g.add_edge("writer", END)

    # 注意：supervisor 没有静态出边——
    # 它通过返回 Command(goto=<节点名>) 在运行时动态决定下一跳。
    return g.compile()


# 全局编译（导入即用）
graph = build_graph()


if __name__ == "__main__":
    from main import main

    main()
