"""
graph.py — LangGraph 工作流编排
将四个节点用有向边串联起来，编译成可执行的工作流。
"""
import os
import sys
# 把当前文件所在目录加入模块搜索路径，使直接 `python graph.py` 也能导入同目录模块
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from state import ResearchState
from nodes import (node_planner, node_researcher, node_critic, node_writer, route_after_critic)

# 加载 .env 环境变量
load_dotenv()

# ============================================================
# 构建图
# ============================================================
def build_graph() -> StateGraph:
    """
    构建并编译 LangGraph StateGraph。

    工作流：
      START → planner → researcher → critic ─┐
                        ↑                      │
                        └── (revision_needed) ─┘
                                   │
                          (not revision_needed)
                                   ↓
                                  writer → END
    """

    # 1. 创建图构建器，指定状态 schema
    builder = StateGraph(ResearchState)

    # 2. 注册四个节点（节点名 → 节点函数）
    builder.add_node("planner",    node_planner)
    builder.add_node("researcher", node_researcher)
    builder.add_node("critic",     node_critic)
    builder.add_node("writer",     node_writer)

    # 3. 添加固定边（执行顺序）
    builder.add_edge(START,           "planner")   # 入口
    builder.add_edge("planner",       "researcher") # 规划完就研究
    builder.add_edge("researcher",    "critic")     # 研究完就审查

    # 4. 添加条件边（Critic 的输出决定下一步）
    #    route_after_critic 返回 "writer" 或 "researcher"
    builder.add_conditional_edges(
        source="critic",
        path=route_after_critic,
        path_map={
            "writer":     END,        # 通过 → 进入写作 → 结束
            "researcher": "researcher",  # 失败 → 打回研究（形成循环）
        },
    )

    # 5. 编译图（生成可执行对象）
    return builder.compile()


# 全局编译（模块加载时执行一次）
graph = build_graph()


# ============================================================
# 可视化：导出图形描述（用于 README / 文档）
# ============================================================
def print_graph_structure():
    """打印图结构的文字描述"""
    print("=" * 60)
    print("Multi-Agent Research Assistant — LangGraph 工作流")
    print("=" * 60)
    print()
    print("  [START] → [planner] → [researcher] → [critic]")
    print("                                       │  ↑")
    print("                              revision_needed")
    print("                                 ↙      ↘")
    print("                         [researcher]  [writer] → [END]")
    print()
    print("  Planner    ：将用户问题拆解为 3~6 个子任务")
    print("  Researcher ：对每个子任务执行 ReAct 搜索循环")
    print("  Critic     ：评审研究质量，决定是否返工")
    print("  Writer     ：整合所有研究，输出结构化报告")
    print()
    print("  条件边逻辑：Critic 评分 < 12 或任一项 < 2 → 打回 Researcher")
    print("             否则 → 进入 Writer → 结束")
    print("  最大循环次数：5 次（防止死循环）")


# ============================================================
# 直接运行入口
# ============================================================
if __name__ == "__main__":
    print_graph_structure()
    print()
    print("-" * 60)
    print("运行示例（使用 Groq 免费 API + Tavily 搜索）")
    print("-" * 60)

    query = input("请输入研究问题：").strip()
    if not query:
        query = "RAG 检索增强生成技术的最新进展"

    print(f"\n正在研究：{query}\n")

    # 带 checkpoint（可选，用于断点续跑）
    # from langgraph.checkpoint.memory import MemorySaver
    # graph = build_graph().compile(checkpointer=MemorySaver())
    # config = {"configurable": {"thread_id": "1"}}
    # result = graph.invoke({"query": query}, config=config)

    result = graph.invoke({"query": query})

    print("\n" + "=" * 60)
    print("最终报告：")
    print("=" * 60)
    print(result.get("final_report", "[无报告]"))

    print("\n" + "-" * 60)
    print("执行统计：")
    print(f"  循环次数：{result.get('loop_count', 0)}")
    print(f"  完成任务：{list(result.get('completed_tasks', {}).keys())}")
    print(f"  评审记录：{len(result.get('critiques', []))} 条")
    print(f"  错误记录：{len(result.get('errors', []))} 条")
