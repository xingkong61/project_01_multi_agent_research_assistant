"""
nodes.py — 四个智能体节点的实现
每个函数是一个 LangGraph 节点，接收当前状态，返回需要更新的状态字段。
"""
import json
import re
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from state import ResearchState

# ============================================================
# 通用设置：LLM 实例（从环境变量读取 API Key）
# ============================================================
def _get_llm(model: str = "groq/llama-3.1-8b-instant") -> ChatOpenAI:
    """
    创建 LLM 实例。
    默认用 Groq（免费），也可切换 OpenAI / Together / DeepSeek。
    传入格式："provider/model-name"，例如 "openai/gpt-4o"
    """
    if model.startswith("groq/"):
        import os
        return ChatOpenAI(
            model=model.split("/", 1)[1],
            api_key=os.getenv("GROQ_API_KEY", ""),
            base_url="https://api.groq.com/openai/v1",
            temperature=0.7,
            max_tokens=4096,
        )
    elif model.startswith("openai/"):
        return ChatOpenAI(model=model.split("/", 1)[1], temperature=0.7, max_tokens=4096)
    else:
        return ChatOpenAI(model="gpt-4o", temperature=0.7, max_tokens=4096)


# ============================================================
# 节点 1：Planner — 任务规划
# ============================================================
def node_planner(state: ResearchState) -> dict:
    """
    Planner 接收用户的原始问题，将其拆解为一组可执行的子任务。
    
    输入状态：state["query"]（用户问题）
    输出状态：state["tasks"]（子任务列表）、state["plan"]（计划说明）、state["loop_count"]=0
    """
    query = state["query"]
    max_loops = state.get("max_loops", 5)

    system_prompt = SystemMessage(content="""你是一个任务规划专家。
给定一个研究问题，你需要将其拆解成 3~6 个互不重叠、可以独立执行的子任务。
每个子任务应该足够具体，便于后续 Researcher 逐一完成。

输出格式（严格遵循 JSON）：
{
  "tasks": ["子任务1", "子任务2", ...],
  "plan": "一句话描述总体研究计划"
}

要求：
- 每个子任务要具体，不要太宽泛（如"了解 RAG 原理"应拆成"RAG 的核心原理"+"RAG 的最新优化方法"）
- 任务之间不要有依赖关系，可以并行执行
- 最多 6 个任务
""")

    llm = _get_llm()
    parser = JsonOutputParser()

    messages = [
        system_prompt,
        HumanMessage(content=f"请拆解以下研究问题：{query}")
    ]

    raw = llm.invoke(messages)
    content = raw.content.strip()

    # 提取 JSON（处理可能的 markdown 包装）
    if "```json" in content:
        content = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL).group(1)
    elif "```" in content:
        content = re.search(r"```\s*(.*?)\s*```", content, re.DOTALL).group(1)

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = {"tasks": [query], "plan": "直接研究"}

    return {
        "tasks": parsed.get("tasks", [query]),
        "plan": parsed.get("plan", ""),
        "loop_count": 0,
        "completed_tasks": {},
        "critiques": [],
        "revision_needed": False,
        "errors": [],
    }


# ============================================================
# 工具封装：Tavily 搜索
# ============================================================
@tool
def _search_web(query: str) -> str:
    """
    封装 Tavily 搜索工具，返回结构化摘要。
    这是 Researcher 智能体在 ReAct 循环中调用的核心工具。
    入参 query 为搜索查询字符串。
    """
    import os
    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        return "[错误] 未设置 TAVILY_API_KEY，请在 .env 中配置"

    tool = TavilySearch(api_key=api_key, max_results=5)
    try:
        results = tool.invoke({"query": query})
        answer_parts = []
        for r in results.get("results", []):
            answer_parts.append(f"来源：{r.get('url', '')}\n标题：{r.get('title', '')}\n摘要：{r.get('content', '')}")
        return "\n\n".join(answer_parts) if answer_parts else "[未找到相关结果]"
    except Exception as e:
        return f"[搜索出错] {str(e)}"


# ============================================================
# 节点 2：Researcher — 研究执行（ReAct 循环）
# ============================================================
def node_researcher(state: ResearchState) -> dict:
    """
    Researcher 对每个未完成的子任务执行 ReAct 推理循环：
    Thought → Action（调用搜索工具）→ Observation → 判断是否继续或结束
    
    当所有任务完成后，输出完整的 completed_tasks 到状态中。
    """
    query = state["query"]
    tasks: list[str] = state.get("tasks", [])
    completed: dict = state.get("completed_tasks", {})

    remaining = [t for t in tasks if t not in completed]

    if not remaining:
        # 所有任务已完成，直接进入 Critic
        return {"completed_tasks": completed}

    current_task = remaining[0]

    # ReAct Prompt
    react_prompt = f"""你是一个专业的研究助手。你正在研究以下主题：

用户原始问题：{query}
当前任务：{current_task}

你的职责是：针对"当前任务"，通过搜索工具收集权威资料。

请按以下步骤执行（ReAct 范式）：

Step 1 - Thought：分析当前任务，决定需要搜索什么关键词或问题。
Step 2 - Action：调用 search_web(query="搜索词") 工具获取资料。
Step 3 - Observation：分析搜索结果，提取与当前任务最相关的核心信息。
Step 4 - 判断：资料是否足够回答当前任务？
  - 如果不够：用新的关键词继续搜索（回到 Step 1）
  - 如果足够：输出研究结论

重要规则：
- 必须至少调用一次 search_web 工具
- 每条结论必须注明来源 URL
- 用中文撰写研究结论（200~400 字）
- 如果搜索失败（如 API 额度不足），仍然可以基于已有信息给出结论，但需注明"信息有限"

开始执行：
"""

    system_msg = SystemMessage(content="你是一个严谨的研究助手，严格遵循 ReAct 范式工作。")
    user_msg = HumanMessage(content=react_prompt)

    # 用新版 bind_tools + tool_calls（langchain 1.x 兼容）
    llm = _get_llm().bind_tools([_search_web])

    messages = [system_msg, user_msg]

    # 最多执行 3 轮 ReAct 循环（防止无限调用）
    for _ in range(3):
        response = llm.invoke(messages)
        ai_msg: AIMessage = response

        # 判断是否调用了工具（新版用 tool_calls 字段）
        if response.tool_calls:
            for tc in response.tool_calls:
                fn_args = tc["args"]
                tool_result = _search_web.invoke(fn_args)

                messages.append(ai_msg)
                messages.append(ToolMessage(content=tool_result, name=tc["name"], tool_call_id=tc["id"]))
        else:
            # LLM 直接给出结论（没有更多工具调用）
            messages.append(ai_msg)
            break

    # 取最后一条 AI 消息作为研究结论
    conclusion = messages[-1].content

    # 提取 URL（简单正则）
    urls = re.findall(r"https?://[^\s）\]\n]+", conclusion)

    completed[current_task] = {
        "content": conclusion,
        "sources": urls,
    }

    return {"completed_tasks": completed}


# ============================================================
# 节点 3：Critic — 评审审查
# ============================================================
def node_critic(state: ResearchState) -> dict:
    """
    Critic 审查已完成任务的质量：
    - 是否所有子任务都有研究结果？
    - 研究结论是否有实质性内容（而非敷衍）？
    - 是否有明显过时或错误的信息？
    
    输出：revision_needed = True（打回 Researcher） 或 False（进入 Writer）
    """
    completed: dict = state.get("completed_tasks", {})
    tasks: list[str] = state.get("tasks", [])
    loop_count: int = state.get("loop_count", 0)
    max_loops = state.get("max_loops", 5)

    # 检查1：是否所有任务都完成了
    missing = [t for t in tasks if t not in completed]
    if missing:
        return {
            "critiques": [f"以下任务尚未完成：{', '.join(missing)}，请继续研究。"],
            "revision_needed": True,
        }

    # 构造审查上下文
    research_summary = "\n\n".join([
        f"【{t}】\n{completed[t]['content']}"
        for t in tasks if t in completed
    ])

    system_prompt = SystemMessage(content="""你是一个严谨的学术评审专家。
你的职责是审查研究结论的质量，判断是否可以进入写作阶段。

评审标准（每项 1-5 分）：
1. 完整性（Completeness）：是否覆盖了任务的所有关键方面？
2. 准确性（Accuracy）：是否存在明显的事实错误或过时信息？
3. 可信度（Credibility）：引用来源是否权威？是否有来源标注？
4. 深度（Depth）：是否仅停留在表面？是否有深入分析？

最终决策：
- 如果总分 >= 12，且没有单项 < 2 → 返回 PASS
- 否则 → 返回 REVISE，并在 critique 中说明具体问题

输出格式（严格 JSON）：
{
  "scores": {"completeness": N, "accuracy": N, "credibility": N, "depth": N},
  "total": N,
  "decision": "PASS" | "REVISE",
  "critique": "如果需要修改，具体指出哪些地方需要加强..."
}
""")

    user_msg = HumanMessage(content=f"请评审以下研究结论：\n\n{research_summary[:4000]}")

    llm = _get_llm()
    parser = JsonOutputParser()
    messages = [system_prompt, user_msg]
    response = llm.invoke(messages)
    content = response.content.strip()

    # 提取 JSON
    if "```json" in content:
        content = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL).group(1)
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        result = {"decision": "PASS", "critique": "", "scores": {}, "total": 12}

    decision = result.get("decision", "PASS")
    critique = result.get("critique", "")

    # 防死循环：如果已达到最大循环次数，强制放行
    if loop_count >= max_loops:
        critique += f"\n[系统提示：已达到最大循环次数 {max_loops}，强制进入写作阶段]"
        decision = "PASS"

    return {
        "critiques": [critique],
        "revision_needed": decision == "REVISE",
        "loop_count": loop_count + 1,
    }


# ============================================================
# 节点 4：Writer — 报告生成
# ============================================================
def node_writer(state: ResearchState) -> dict:
    """
    Writer 整合所有研究结果，生成结构化的最终报告。
    强制要求：
    - 每个章节必须标注信息来源
    - 不得凭空捏造信息
    - 格式：标题 → 摘要 → 各章节 → 参考资料
    """
    query = state["query"]
    plan = state.get("plan", "")
    tasks: list[str] = state.get("tasks", [])
    completed: dict = state.get("completed_tasks", {})

    research_texts = []
    for task in tasks:
        if task in completed:
            src = completed[task]
            research_texts.append(f"## {task}\n\n{src['content']}\n\n来源：{'；'.join(src['sources']) if src['sources'] else '无'}")

    research_context = "\n\n".join(research_texts)

    system_prompt = SystemMessage(content="""你是一个专业的研究报告写作专家。

写作要求（严格遵守）：
1. 结构：标题 → 摘要（150字以内）→ 各章节 → 参考资料
2. 每个章节内容必须基于提供的参考资料撰写，绝不能凭空编造
3. 每个重要结论后必须标注来源 URL
4. 如果某部分参考资料不足，明确说明"资料有限，以下内容基于现有研究"
5. 语言专业、简洁，适合技术从业者阅读
6. 最终报告应全面覆盖用户的研究问题

参考资料：
{context}
""".format(context=research_context))

    user_msg = HumanMessage(content=f"请根据以上研究资料，为以下问题撰写完整报告：{query}")

    llm = _get_llm()
    messages = [system_prompt, user_msg]
    response = llm.invoke(messages)

    return {"final_report": response.content}


# ============================================================
# 路由器：Critic → 下一步走哪里
# ============================================================
def route_after_critic(state: ResearchState) -> str:
    """
    条件边函数：Critic 评审后，决定下一步走哪个节点。
    - revision_needed == True → 返回 "researcher"（打回重新研究）
    - revision_needed == False → 返回 "writer"（进入写作）
    """
    if state.get("revision_needed", False):
        return "researcher"
    return "writer"
