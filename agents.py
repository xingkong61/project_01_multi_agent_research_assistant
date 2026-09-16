"""
agents.py — 五个智能体节点的实现

架构：Supervisor 模式（LLM 动态调度 + 规则护栏）

  START → supervisor ──LLM 在运行时决定 handoff──→ planner / researcher / critic / writer
              ↑                                                                    │
              └──────────────── 每个 worker 完成后回到 supervisor ←─────────────────┘
                                                                              writer → END

与旧版"固定流水线"的本质区别：
1. 控制流不再写死：Supervisor 是一个 LLM 节点，每步基于黑板状态选择下一个智能体，
   代码只做合法性校验与兜底（_allowed_next / _rule_based_next）。
2. Researcher 是真正的工具型智能体：3 个真实工具（网页搜索 / 网页抓取 / arXiv），
   自己决定搜什么、用哪个工具、搜几轮，并通过终态工具 submit_finding 自主宣布完成。
3. 反馈闭环真正接通：Critic 的 issues 会进入 Researcher 下一轮 prompt，
   不合格任务被重置为 pending，研究结论按 revision 迭代。
4. Planner 支持 replan：当 Critic 认为题目本身没拆全（REPLAN），
   Planner 会基于已有结论与缺失主题补充子任务，而不是拆一次就固定。
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langgraph.types import Command, Overwrite

from llm import get_llm
from state import Critique, Finding, ResearchState, Source, Task
from tools import RESEARCH_TOOLS

# 每个 researcher 节点内部最多的 ReAct 步数
MAX_REACT_STEPS = 12  # 推理类模型思考耗时长，单步可能只做一次检索，步数适当放宽
# 传给模型的单条工具结果长度上限
_TOOL_RESULT_LIMIT = 12000


# ============================================================
# 通用工具函数
# ============================================================
def _text(message: AIMessage) -> str:
    """兼容 content 为字符串或结构化块两种形态。"""
    content = message.content
    return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)


def _finish_reason(message: AIMessage) -> str:
    """提取 provider 返回的结束原因：'length' 表示输出被 max_tokens 截断。"""
    meta = getattr(message, "response_metadata", None) or {}
    return str(meta.get("finish_reason", "")).lower()


# 结尾标点（中英文）：正常写完的结论几乎必然以其中之一收尾
_SENTENCE_END = tuple("。！？!?…\"'”’)】》>.")


def _looks_truncated(text: str) -> bool:
    """启发式判断一段文本是否被物理截断（句子中间戛然而止）。

    仅作为 finish_reason 不可用时的补充信号：正文足够长、且末尾不是任何
    句末标点时，判定为疑似截断。刻意保守——宁可漏判也不误伤正常短结论。
    """
    t = text.strip()
    if len(t) < 60:            # 太短不可能是"写到一半"
        return False
    # 允许末尾带 Markdown 代码围栏闭合等符号
    t = t.rstrip("`* \n\t")
    if not t:
        return False
    return not t.endswith(_SENTENCE_END)


def describe_exc(exc: BaseException) -> str:
    """把 LLM/网络异常翻译成排障者真正需要的信息。

    openai SDK 的 APIStatusError（429 限流 / 401 鉴权 / 内容审核等）直接 f"{exc}"
    常常丢失服务端返回的具体 message，导致界面上只剩"调用失败"却看不到根因。
    这里尽力挖出 HTTP 状态码与响应体里的 error.message，并给出常见错误的中文提示。
    """
    parts: list[str] = [type(exc).__name__]

    status = getattr(exc, "status_code", None)
    resp = getattr(exc, "response", None)
    if status is None and resp is not None:
        status = getattr(resp, "status_code", None)
    if status is not None:
        parts.append(f"HTTP {status}")

    # 尝试从响应体提取服务端错误说明
    body_msg = ""
    try:
        import json as _json
        text = None
        if resp is not None and hasattr(resp, "text"):
            text = resp.text
        elif hasattr(exc, "body") and exc.body is not None:
            text = _json.dumps(exc.body, ensure_ascii=False)
        if text:
            try:
                data = _json.loads(text)
                err = data.get("error") if isinstance(data, dict) else None
                body_msg = (err or {}).get("message", "") if isinstance(err, dict) else str(data)[:200]
            except Exception:
                body_msg = text[:200]
    except Exception:
        pass
    if body_msg:
        parts.append(body_msg.strip())
    else:
        msg = str(exc).strip()
        if msg:
            parts.append(msg[:200])

    desc = " | ".join(parts)

    # 针对高频根因给一句可操作的中文提示
    low = desc.lower()
    hint = ""
    if status == 429 or "rate limit" in low or "quota" in low or "too many" in low:
        hint = "（触发限流/额度上限：稍后重试、降低并发或提升配额）"
    elif status in (401, 403) or "invalid api key" in low or "authentication" in low:
        hint = "（API Key 无效或无权限：检查 .env 中的 LLM_API_KEY）"
    elif "timeout" in low or "timed out" in low:
        hint = "（请求超时：可提高 LLM_TIMEOUT 或换更快的模型）"
    elif "model" in low and ("not found" in low or "does not exist" in low or "not exist" in low):
        hint = "（模型名不存在：核对 LLM_MODEL 是否为该端点支持的型号）"
    elif "content" in low and ("policy" in low or "review" in low or "safety" in low):
        hint = "（内容安全审核拦截：调整问题措辞）"
    return (desc + " " + hint).strip()


def _extract_json(content: str) -> dict | None:
    """从模型输出中提取 JSON，兼容 ```json 代码块包裹。失败返回 None。"""
    text = content.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        # 再抢救一次：截取第一个 { 到最后一个 }
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                data = json.loads(text[start:end + 1])
                return data if isinstance(data, dict) else None
            except json.JSONDecodeError:
                return None
    return None


def normalize_sources(raw: Any) -> list[Source]:
    """
    把模型提交的 sources（可能是 URL 字符串列表 / 字典列表 / 空值）
    归一化为 Source 列表，并按 URL 去重。纯函数，可单测。
    """
    sources: list[Source] = []
    seen: set[str] = set()

    def _add(url: str, title: str = "") -> None:
        url = (url or "").strip().rstrip(".,;)")
        if not url.startswith(("http://", "https://")):
            return
        if url in seen:
            return
        seen.add(url)
        sources.append({"url": url, "title": title.strip()})

    if isinstance(raw, str):
        for u in re.findall(r"https?://[^\s）\]\"'<>]+", raw):
            _add(u)
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, str):
                _add(item)
            elif isinstance(item, dict):
                _add(str(item.get("url", "")), str(item.get("title", "")))

    return sources


# ------------------------------------------------------------
# 引用可信度校验：识别模型编造的链接（尤其伪造的 arXiv ID）
# ------------------------------------------------------------
_ARXIV_ID_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4})\.(\d{4,5})", re.IGNORECASE)


def is_implausible_arxiv_id(url: str, now=None) -> bool:
    """判断 URL 中的 arXiv 编号是否为不可能存在的"未来/伪造"ID。

    arXiv 新编号格式 YYNN.nnnnn：YY 为年份后两位、NN 为月份 01~12。
    模型幻觉时常编出超出当前年月或月份非法的编号（如 2613.xxxxx、9999.xxxxx）。
    返回 True 表示该来源高度可疑，应剔除或降级。纯函数，可离线单测。
    """
    from datetime import datetime

    m = _ARXIV_ID_RE.search(url or "")
    if not m:
        return False                       # 非 arXiv 链接，不在此判定
    yy_str, nn_str = m.group(1), m.group(2)
    year = 2000 + int(yy_str[:2])
    month = int(yy_str[2:]) if len(yy_str) >= 4 else 0
    # 旧式编号（如 cs/0112017）走不到这里；此处只处理新式 YYMM
    if not (1 <= month <= 12):
        return True                        # 月份非法 → 伪造
    ref_year, ref_month = (now.year, now.month) if now else (datetime.now().year, datetime.now().month)
    # 论文编号年月不应晚于"下个月"（容忍极小时钟/发布偏差）。用绝对月序号比较，避免跨年进位错误。
    return year * 12 + month > ref_year * 12 + ref_month + 1


def validate_citations(sources: list[Source], observed_urls: set[str], now=None) -> tuple[list[Source], list[str]]:
    """交叉校验引用来源，返回 (可信来源列表, 被剔除原因列表)。

    两条规则：
    1) 溯源：URL 必须真实出现在本轮某个工具返回里（observed_urls），
       否则视为模型凭空编造，剔除——落实 prompt「只能引用工具返回中真实出现过的 URL」。
    2) 合理性：即便出现过，arXiv 编号若是"未来/非法"也剔除（防模型改写真实链接时手滑）。
    无 observed_urls（None）时跳过溯源、仅做合理性校验，保持向后兼容。
    """
    kept: list[Source] = []
    dropped: list[str] = []
    obs = {u.rstrip(".,;)") for u in (observed_urls or set())}
    for s in sources:
        url = (s.get("url") or "").strip().rstrip(".,;)")
        if is_implausible_arxiv_id(url, now=now):
            dropped.append(f"{url}（arXiv 编号疑似伪造/未来日期）")
            continue
        if obs and url not in obs:
            dropped.append(f"{url}（未在任何检索结果中出现，疑似编造）")
            continue
        kept.append(s)
    return kept, dropped


def _task_index(tasks: list[Task]) -> int:
    """下一个新任务的序号（兼容 t1/t2 命名）。"""
    biggest = 0
    for t in tasks:
        m = re.fullmatch(r"t(\d+)", str(t.get("id", "")))
        if m:
            biggest = max(biggest, int(m.group(1)))
    return biggest + 1


def _trace(msg: str) -> dict:
    return {"trace": [msg]}


# ============================================================
# 节点 1：Planner —— 初始规划 / 基于评审意见重新规划
# ============================================================
_PLANNER_SCHEMA_HINT = """严格输出 JSON（不要输出 JSON 以外的内容）：
{
  "plan": "一句话总体研究思路",
  "tasks": [
    {"description": "具体的子任务描述", "rationale": "该任务对回答用户问题的作用"}
  ]
}"""


def _planner_prompt(state: ResearchState, replan: bool) -> str:
    query = state["query"]
    if not replan:
        return f"""研究问题：{query}

请把问题拆解为 3~5 个互不重叠、可以独立调研的子任务。要求：
- 具体可执行，避免"了解 XX 原理"这种过宽描述，应拆成"核心原理""关键方法""最新进展"等
- 覆盖：背景与定义、核心方法/机制、最新进展（近 1~2 年）、争议或局限、应用与趋势
- 对偏学术的问题优先考虑可在 arXiv 检索的主题；对偏时效的问题优先考虑联网搜索
- 子任务之间尽量正交，数量不超过 5 个

{_PLANNER_SCHEMA_HINT}"""

    # replan：保留已完成任务，只补充缺口
    critique = state.get("critique") or {}
    missing = critique.get("missing_topics", [])
    existing_desc = [t["description"] for t in state.get("tasks", [])]
    findings = state.get("findings", {})
    covered = "\n".join(f"- {d}" for d in existing_desc)
    gaps = "\n".join(f"- {m}" for m in missing) or "- （评审未列出具体缺口，请自行判断盲区）"
    return f"""研究问题：{query}

已有子任务（均已完成，请不要重复）：
{covered}

评审认为还缺少以下方向：
{gaps}

请只补充 1~3 个**新增**子任务来填补缺口（不要重复已有任务）。
{_PLANNER_SCHEMA_HINT}"""


def node_planner(state: ResearchState) -> dict:
    """初始规划，或在 Critic 给出 REPLAN 后补充子任务。"""
    replan = bool(state.get("tasks"))
    try:
        llm = get_llm(temperature=0.3)
        resp = llm.invoke([
            SystemMessage(content="你是研究项目的负责人，擅长把模糊问题拆解成可执行的研究计划。"),
            HumanMessage(content=_planner_prompt(state, replan)),
        ])
        parsed = _extract_json(_text(resp)) or {}
        raw_tasks = parsed.get("tasks", [])
        plan = str(parsed.get("plan", "")).strip()
    except Exception as exc:  # Key/网络/额度问题：兜底，保证流程不断
        raw_tasks, plan = [], ""
        errors = [f"Planner 调用失败，使用兜底计划：{describe_exc(exc)}"]
    else:
        errors = []

    tasks: list[Task] = list(state.get("tasks", []))
    existing_lower = {t["description"].strip().lower() for t in tasks}
    new_items: list[Task] = []

    idx = _task_index(tasks)
    for item in raw_tasks:
        if isinstance(item, str):
            desc, rationale = item.strip(), ""
        elif isinstance(item, dict):
            desc = str(item.get("description", "")).strip()
            rationale = str(item.get("rationale", "")).strip()
        else:
            continue
        if not desc or desc.lower() in existing_lower:
            continue
        existing_lower.add(desc.lower())
        new_items.append({
            "id": f"t{idx}",
            "description": desc,
            "rationale": rationale,
            "status": "pending",
        })
        idx += 1

    # 模型没给出任何任务时的兜底
    if not tasks and not new_items:
        new_items = [{
            "id": "t1",
            "description": state["query"],
            "rationale": "规划失败，直接研究原始问题",
            "status": "pending",
        }]
        plan = plan or "直接研究原始问题"

    update: dict = {
        "tasks": tasks + new_items,
        "trace": [f"Planner：{'补充' if replan else '制定'} {len(new_items)} 个子任务"
                  + (f"（{plan}）" if plan else "")],
    }
    if not replan:
        update["plan"] = plan
        update["iteration"] = 0
        update["findings"] = {}
        # critique_history 是 operator.add reducer，返回 [] 只会"追加空列表"、
        # 无法清空旧值；必须用 Overwrite 显式覆盖，才能真正重置评审历史。
        update["critique_history"] = Overwrite([])
        update["errors"] = errors
    else:
        update["trace"][0] += f"；新增：{'；'.join(t['description'] for t in new_items)}"
        if errors:
            update["errors"] = errors
        # 旧 REPLAN 结论必须解除：有新任务则做完后复审；
        # 一个都没补出来也不能无限补题，直接回到 Critic 按现有材料评审。
        update["critique"] = {
            "decision": "REVIEW",
            "scores": {},
            "issues": [],
            "missing_topics": [],
            "comment": ("已补充新子任务，等待研究与评审。" if new_items
                        else "Planner 未能补充新任务，按现有材料继续。"),
        }
    return update


# ============================================================
# 节点 2：Researcher —— 真正的工具调用型智能体（ReAct）
# ============================================================
def _select_target(state: ResearchState) -> Task:
    """挑一个待研究任务：优先返工评审点名的任务。"""
    tasks = state.get("tasks", [])
    pending = [t for t in tasks if t.get("status") != "done"]
    critique = state.get("critique") or {}
    flagged = {
        i.get("task_id")
        for i in critique.get("issues", [])
        if i.get("task_id")
    }
    for t in pending:
        if t["id"] in flagged:
            return t
    return pending[0]


def _build_feedback(state: ResearchState, target: Task) -> str:
    """把 Critic 的评审意见整理成给 Researcher 的返工要求。"""
    critique = state.get("critique") or {}
    if not critique:
        return ""
    lines = []
    for issue in critique.get("issues", []):
        if not issue.get("task_id") or issue["task_id"] == target["id"]:
            lines.append(f"- {issue.get('issue', '')}")
    comment = critique.get("comment", "")
    if comment:
        lines.append(f"- 总体评语：{comment}")
    if not lines:
        return ""
    return ("以下是上一轮评审提出的问题，本轮必须针对性补强：\n"
            + "\n".join(lines))


def node_researcher(state: ResearchState) -> dict:
    """
    ReAct 研究智能体（一次处理一个任务）：
    Thought → 自主选择工具调用 → Observation → ... → submit_finding 自主结束
    """
    tasks: list[Task] = list(state.get("tasks", []))
    findings: dict[str, Finding] = dict(state.get("findings", {}))

    pending = [t for t in tasks if t.get("status") != "done"]
    if not pending:
        return _trace("Researcher：没有待研究任务，空转返回")

    target = _select_target(state)
    feedback = _build_feedback(state, target)
    prior = findings.get(target["id"])
    revision = (prior or {}).get("revision", 0) + 1

    # 其他任务的结论摘要（避免重复劳动）
    others = [
        f"- [{fid}] {findings[fid].get('summary', '')[:120]}"
        for fid in findings if fid != target["id"]
    ]
    others_text = "\n".join(others)

    # ---------- 终态工具：由模型自己决定"资料够了" ----------
    @tool
    def submit_finding(summary: str, sources: list, confidence: str = "medium") -> str:
        """提交当前任务的最终研究结论并结束本任务。当且仅当你认为资料已经充分时调用。

        参数：
        - summary: 中文研究结论，250~600 字，Markdown，关键事实后用 (来源: URL) 标注
        - sources: 本次结论实际引用的来源列表，每项形如 {"title": "标题", "url": "https://..."}
        - confidence: 结论可信度，high / medium / low 之一（资料不足时用 low）
        """
        return "结论已提交。"

    tool_map = {t.name: t for t in RESEARCH_TOOLS}
    all_tools = RESEARCH_TOOLS + [submit_finding]

    system_prompt = SystemMessage(content=(
        "你是为研究人员服务的资深研究助手，通过工具自行收集资料。"
        "工作方式：先思考需要什么信息 → 选择合适的工具 → 阅读返回结果 → "
        "判断是否足够 → 不足则换关键词/换工具继续，足够则调用 submit_finding 提交。"
        "铁律：(1) 至少调用一次研究工具，禁止凭记忆作答；"
        "(2) 只能引用工具返回中真实出现过的 URL，禁止编造链接；"
        "(3) 搜索关键词优先使用英文，学术主题优先 search_arxiv，"
        "需要阅读网页细节时用 fetch_webpage，时效性主题用 web_search；"
        "(4) 工具报错时换一种方式继续，不要把错误信息写进结论；"
        "(5) 检索预算：全部工具合计不超过 6 次，用完或认为资料足够时"
        "必须立即调用 submit_finding 提交——资料不足就如实给低可信度结论，"
        "禁止无限检索、禁止用尽步数被强制收尾；"
        "(6) 结论使用中文。"
    ))

    user_prompt = HumanMessage(content=f"""研究问题：{state['query']}
你的任务（ID={target['id']}）：{target['description']}
任务意义：{target.get('rationale') or '（无）'}
{('这是第 ' + str(revision) + ' 轮研究。') if revision > 1 else ''}
{feedback}
{'其他任务已覆盖的内容（不要重复）：' + chr(10) + others_text if others_text else ''}

请开始：选择工具、收集资料，最后调用 submit_finding 提交结论。""")

    messages = [system_prompt, user_prompt]

    submitted: dict | None = None
    last_text = ""
    stats = {"searches": 0, "pages": 0}
    nudged = False
    run_errors: list[str] = []
    # 同参工具调用结果缓存：模型常对相同关键词重复检索（实测一轮内连发 2~3 次
    # 相同的 search_arxiv/web_search），重复请求白烧 Tavily 额度、白等 arXiv 限流
    tool_cache: dict[str, str] = {}
    # 本轮所有工具返回里真实出现过的 URL，用于引用溯源校验（剔除模型编造的链接）
    observed_urls: set[str] = set()

    # 结论被 max_tokens 截断时，加大输出预算重跑提交的上限与当前预算。
    # 根因：qwen3 等推理模型的"思考 token"与正文一起计入 max_tokens，ReAct
    # 后期上下文很大时，写结论的预算被思考挤占 → 半句被砍 → Critic 反复打回同一
    # 任务、评审空转到耗尽轮数。检测到截断就放大预算重试，而不是提交残缺结论。
    _BASE_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "8192"))
    _TRUNCATION_RETRIES = 2
    cur_max_tokens = _BASE_MAX_TOKENS
    truncation_retry = 0

    try:
        bound = get_llm(temperature=0.4, max_tokens=cur_max_tokens).bind_tools(all_tools)
        for _ in range(MAX_REACT_STEPS):
            ai: AIMessage = bound.invoke(messages)
            messages.append(ai)
            if _text(ai).strip():
                last_text = _text(ai).strip()

            if not ai.tool_calls:
                # 模型没有调用工具：提醒一次，仍不调用就收尾
                if submitted is not None:
                    break
                if nudged:
                    break
                messages.append(HumanMessage(
                    "你还没有结束本任务。若资料已充分，请立即调用 submit_finding 提交结论；"
                    "若资料不足，请继续调用研究工具。"
                ))
                nudged = True
                continue
            nudged = False

            cut = _finish_reason(ai) == "length"   # provider 权威信号：输出被截断
            for tc in ai.tool_calls:
                name, args = tc.get("name", ""), tc.get("args") or {}
                if name == "submit_finding":
                    summary_txt = str((args or {}).get("summary", "")) if isinstance(args, dict) else ""
                    # 判定本次提交是否残缺：provider 报 length，或结尾疑似被切断
                    truncated = cut or (bool(summary_txt) and _looks_truncated(summary_txt))
                    if truncated and truncation_retry < _TRUNCATION_RETRIES:
                        # 不接受残缺结论：放大输出预算，要求模型重新完整提交
                        truncation_retry += 1
                        cur_max_tokens *= 2
                        run_errors.append(
                            f"Researcher({target['id']}) 第 {revision} 版结论被截断"
                            f"（finish_reason={'length' if cut else 'heuristic'}），"
                            f"提高 max_tokens 至 {cur_max_tokens} 后重试提交")
                        bound = get_llm(temperature=0.4, max_tokens=cur_max_tokens).bind_tools(all_tools)
                        messages.append(ToolMessage(
                            "你上一次提交的结论在句子中间被截断了，内容不完整。"
                            "请基于已有资料，一次性输出**完整**的研究结论并重新调用 submit_finding。",
                            tool_call_id=tc.get("id", ""), name=name))
                        break
                    submitted = args if isinstance(args, dict) else {}
                    messages.append(ToolMessage(
                        "结论已提交，本任务结束。", tool_call_id=tc.get("id", ""), name=name))
                    break

                research_tool = tool_map.get(name)
                if research_tool is None:
                    result = f"[工具错误] 未知工具：{name}"
                else:
                    # 统计口径：按"模型发起的研究工具调用次数"计（含命中缓存的重复检索）。
                    # 以工具身份而非硬编码名称分类，保证自定义/测试替换的工具也被正确计数。
                    if name == "fetch_webpage":
                        stats["pages"] += 1
                    else:
                        stats["searches"] += 1
                    # 同参数调用命中缓存：不重复请求（省 Tavily 额度 / 避开 arXiv 限流等待）
                    cache_key = f"{name}:{json.dumps(args, ensure_ascii=False, sort_keys=True)}"
                    cached = tool_cache.get(cache_key)
                    if cached is not None:
                        result = cached + "\n[与此前请求完全相同，已直接复用结果，请勿重复检索同一关键词]"
                    else:
                        try:
                            result = str(research_tool.invoke(args))
                        except Exception as exc:
                            result = f"[工具错误] {name} 执行失败：{exc}"
                        tool_cache[cache_key] = result
                # 记录该工具返回中真实出现的 URL，供后续引用溯源（含缓存命中结果）
                for u in re.findall(r"https?://[^\s)\]）\"'<>,，；;。，]+", result):
                    observed_urls.add(u.rstrip(".,)。；;：:"))
                messages.append(ToolMessage(
                    result[:_TOOL_RESULT_LIMIT],
                    tool_call_id=tc.get("id", ""),
                    name=name,
                ))
            if submitted is not None:
                break
    except Exception as exc:
        run_errors.append(f"Researcher({target['id']}) 推理循环异常：{describe_exc(exc)}")

    # ---------- 归一化产出 ----------
    if submitted and str(submitted.get("summary", "")).strip():
        summary = str(submitted["summary"]).strip()
        sources = normalize_sources(submitted.get("sources"))
        confidence = str(submitted.get("confidence", "medium")).lower()
        if confidence not in ("high", "medium", "low"):
            confidence = "medium"
    else:
        # 模型始终未提交：用最后一段文本兜底，可信度调低
        summary = last_text or "（智能体未能产出结论：工具调用失败或模型未按要求提交）"
        sources = normalize_sources(summary)
        confidence = "low"
        run_errors.append(f"Researcher({target['id']}) 未调用 submit_finding，使用兜底结论")

    # 结论里提到但 sources 漏报的 URL 补录
    known = {s["url"] for s in sources}
    for u in re.findall(r"https?://[^\s)\]）\"'<>,，；;。]+", summary):
        u = u.rstrip(".,)。；;：:")
        if u and u not in known:
            sources.append({"url": u, "title": ""})
            known.add(u)

    # ---------- 引用可信度校验：剔除伪造/无出处链接 ----------
    # 仅在本轮确有工具返回 URL 时启用溯源（observed_urls 非空），避免检索全失败时误杀。
    kept, dropped = validate_citations(sources, observed_urls or None)
    if dropped:
        run_errors.append(
            f"Researcher({target['id']}) 剔除了 {len(dropped)} 条不可信引用："
            + "；".join(dropped[:3]))
        # 有来源被证伪 → 结论可信度下调一档（high→medium→low）
        confidence = {"high": "medium", "medium": "low", "low": "low"}[confidence]
    sources = kept

    findings[target["id"]] = {
        "task_id": target["id"],
        "summary": summary,
        "sources": sources,
        "confidence": confidence,
        "searches": stats["searches"],
        "pages_fetched": stats["pages"],
        "revision": revision,
    }
    for t in tasks:
        if t["id"] == target["id"]:
            t["status"] = "done"

    update = {
        "tasks": tasks,
        "findings": findings,
        "trace": [
            f"Researcher：完成任务 {target['id']}「{target['description'][:24]}…」"
            f"（第 {revision} 版，搜索 {stats['searches']} 次，抓取 {stats['pages']} 次，"
            f"来源 {len(sources)} 条，可信度 {confidence}）"
        ],
        "errors": run_errors,
    }

    # 返工任务已全部做完：把评审状态置为 REVIEW，通知 Supervisor 重新评审，
    # 而不是拿着旧的 REVISE 结论继续派活。
    prior_critique = state.get("critique") or {}
    still_pending = [t for t in tasks if t.get("status") != "done"]
    if prior_critique.get("decision") == "REVISE" and not still_pending:
        update["critique"] = {
            "decision": "REVIEW",
            "scores": {},
            "issues": [],
            "missing_topics": [],
            "comment": "返工已完成，等待 Critic 复审。",
        }
    return update


# ============================================================
# 节点 3：Critic —— LLM 评审 + 代码级完整性闸门
# ============================================================
def _dossier(state: ResearchState, limit: int = 1800) -> str:
    tasks = state.get("tasks", [])
    findings = state.get("findings", {})
    blocks = []
    for t in tasks:
        f = findings.get(t["id"])
        if f:
            src_urls = "; ".join(s.get("url", "") for s in f.get("sources", []))
            blocks.append(
                f"### 任务 {t['id']}：{t['description']}\n"
                f"可信度（研究员自评）：{f.get('confidence')}\n"
                f"结论：\n{f.get('summary', '')[:limit]}\n"
                f"来源：{src_urls or '无'}"
            )
        else:
            blocks.append(f"### 任务 {t['id']}：{t['description']}\n【缺失：没有研究结论】")
    return "\n\n".join(blocks)


def node_critic(state: ResearchState) -> dict:
    """评审研究质量，输出 PASS / REVISE（补强）/ REPLAN（补题）。"""
    tasks: list[Task] = list(state.get("tasks", []))
    findings = state.get("findings", {})
    iteration = state.get("iteration", 0)
    errors: list[str] = []

    # —— 代码级闸门：有任务根本没结论，直接返工，不浪费一次 LLM 评审 ——
    missing = [t for t in tasks if t["id"] not in findings]
    if missing:
        critique: Critique = {
            "decision": "REVISE",
            "scores": {},
            "issues": [{"task_id": t["id"], "issue": "缺少研究结论，请完成该任务。"}
                       for t in missing],
            "missing_topics": [],
            "comment": "存在未完成的任务。",
        }
    else:
        prompt = f"""请评审以下研究材料是否足以支撑撰写一份给研究人员看的报告。

评分维度（每项 1~5 分）：
- completeness：是否完整覆盖该任务，没有关键缺口
- accuracy：事实是否准确、有无明显错误或过时信息
- credibility：是否有真实、权威、可追溯的来源支撑（无来源直接打 2 分以下）
- depth：是否有足够深度，而非泛泛而谈

决策规则：
- 总分 >= 15 且无单项 <= 2 → "PASS"
- 材料质量不达标、某些任务需要补充检索/深化 → "REVISE"，
  并在 issues 中逐条指出（必须尽量带上对应任务 ID）
- 发现研究计划本身漏了重要方向、需要新增子任务 → "REPLAN"，
  并在 missing_topics 中列出建议新增的主题

严格输出 JSON：
{{
  "scores": {{"completeness": N, "accuracy": N, "credibility": N, "depth": N}},
  "decision": "PASS" 或 "REVISE" 或 "REPLAN",
  "issues": [{{"task_id": "t1", "issue": "具体问题与补强方向"}}],
  "missing_topics": ["需要新增的主题"],
  "comment": "总体评语"
}}

研究材料：
{_dossier(state)}"""
        try:
            llm = get_llm(temperature=0.2, thinking=False)
            resp = llm.invoke([
                SystemMessage(content="你是严格的学术评审，只依据材料本身判断，不脑补来源。"),
                HumanMessage(content=prompt),
            ])
            data = _extract_json(_text(resp))
        except Exception as exc:
            data = None
            errors.append(f"Critic 调用失败，按通过处理：{describe_exc(exc)}")

        if not data:
            critique = {
                "decision": "PASS",
                "scores": {},
                "issues": [],
                "missing_topics": [],
                "comment": "评审模型不可用，系统按通过处理。",
            }
        else:
            scores = data.get("scores") if isinstance(data.get("scores"), dict) else {}
            cleaned: dict[str, int] = {}
            for k, v in scores.items():
                # 模型偶尔输出 "3.5" / "4.0" 这类字符串小数，str(v).isdigit() 会误判丢弃；
                # 统一走 float 解析再夹到 [1,5]，非法值（None/文本/布尔）跳过。
                if isinstance(v, bool):
                    continue
                try:
                    cleaned[k] = max(1, min(5, int(round(float(v)))))
                except (TypeError, ValueError):
                    continue
            scores = cleaned
            issues = []
            for i in data.get("issues", []) if isinstance(data.get("issues"), list) else []:
                if isinstance(i, dict):
                    issues.append({
                        "task_id": str(i.get("task_id", "")) or "",
                        "issue": str(i.get("issue", "")).strip(),
                    })
            missing_topics = [
                str(x).strip() for x in (data.get("missing_topics") or [])
                if str(x).strip()
            ]
            decision = str(data.get("decision", "PASS")).upper()
            if decision not in ("PASS", "REVISE", "REPLAN"):
                decision = "PASS"
            critique = {
                "decision": decision,
                "scores": scores,
                "issues": [i for i in issues if i["issue"]],
                "missing_topics": missing_topics,
                "comment": str(data.get("comment", "")).strip(),
            }

    # —— 根据决策更新任务状态 ——
    update: dict = {
        "critique": critique,
        "critique_history": [critique],
        "iteration": iteration + 1,
        "trace": [],
        "errors": errors,
    }

    if critique["decision"] == "REVISE":
        flagged = {i["task_id"] for i in critique["issues"] if i.get("task_id")}
        reset = []
        for t in tasks:
            if not flagged or t["id"] in flagged:
                if t.get("status") == "done":
                    t["status"] = "pending"
                    reset.append(t["id"])
        update["tasks"] = tasks
        score_text = (
            f"，得分 {sum(critique['scores'].values())}/20"
            if critique.get("scores") else ""
        )
        update["trace"] = [
            f"Critic：第 {iteration + 1} 轮评审 → REVISE{score_text}，"
            f"打回任务 {reset or '（全部）'}："
            + "；".join(i["issue"][:60] for i in critique["issues"][:3])
        ]
    elif critique["decision"] == "REPLAN":
        update["trace"] = [
            f"Critic：第 {iteration + 1} 轮评审 → REPLAN，建议补充方向："
            + "；".join(critique["missing_topics"][:3])
        ]
    else:
        total = sum(critique.get("scores", {}).values())
        update["trace"] = [
            f"Critic：第 {iteration + 1} 轮评审 → PASS"
            + (f"（{total}/20）" if total else "")
        ]

    return update


# ============================================================
# 节点 4：Writer —— 汇总成稿（永远产出报告，含确定性兜底）
# ============================================================
def _fallback_report(query: str, tasks: list[Task], findings: dict[str, Finding]) -> str:
    """LLM 不可用时的确定性报告：直接汇编原始研究结论，保证用户有产出。"""
    lines = [f"# {query}", "", "> 注：写作模型当前不可用，以下为研究结论的自动汇编版。", ""]
    for t in tasks:
        f = findings.get(t["id"])
        lines.append(f"## {t['description']}")
        lines.append("")
        lines.append(f.get("summary", "（无结论）") if f else "（无结论）")
        lines.append("")
        if f and f.get("sources"):
            lines.append("**参考来源：**")
            for s in f["sources"]:
                lines.append(f"- [{s.get('title') or s['url']}]({s['url']})")
            lines.append("")
    return "\n".join(lines)


def node_writer(state: ResearchState) -> dict:
    query = state["query"]
    tasks = state.get("tasks", [])
    findings = state.get("findings", {})

    try:
        # Writer 做的是"材料加工"而非推理规划：关闭深度思考，
        # 避免思考耗尽 token 预算导致空响应，同时大幅降低耗时与 token 成本
        resp = get_llm(temperature=0.4, thinking=False).invoke([
            SystemMessage(content=(
                "你是研究报告写作专家。只依据提供的材料写作，严禁编造事实或链接。"
                "报告结构：# 标题 → ## 摘要（150 字以内）→ 若干正文章节 → ## 参考资料。"
                "要求：(1) 每个关键事实后用 Markdown 链接标注来源，如 [标题](URL)；"
                "(2) 材料不足处明确写'资料有限'，不要掩盖；"
                "(3) 中文撰写，专业、简洁、有信息量；"
                "(4) 参考资料按章节出现顺序编号去重列出。"
            )),
            HumanMessage(content=(
                f"研究问题：{query}\n\n研究材料：\n{_dossier(state, limit=4000)}\n\n"
                "请撰写完整的 Markdown 研究报告。"
            )),
        ])
        report = _text(resp).strip()
        if not report:
            raise ValueError("模型返回为空")
    except Exception as exc:
        report = _fallback_report(query, tasks, findings)
        return {
            "report_markdown": report,
            "errors": [f"Writer 调用失败，使用汇编兜底：{describe_exc(exc)}"],
            "trace": ["Writer：模型不可用，已生成汇编版报告"],
        }

    return {
        "report_markdown": report,
        "trace": ["Writer：已生成最终研究报告"],
    }


# ============================================================
# 节点 5：Supervisor —— LLM 运行时动态调度（规则护栏兜底）
# ============================================================
def _allowed_next(state: ResearchState) -> list[str]:
    """
    根据当前黑板状态计算合法 handoff 集合。
    大多数状态下只有一个合法目标；存在真实选择空间时才让 LLM 决断。
    """
    tasks = state.get("tasks") or []
    findings = state.get("findings") or {}
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 4)

    # 预算耗尽：强制收尾（前提是已有研究产出）
    if iteration >= max_iterations and findings:
        return ["writer"]
    if not tasks:
        return ["planner"]

    pending = [t for t in tasks if t.get("status") != "done"]
    critique = state.get("critique")

    if pending:
        if critique and critique.get("decision") == "REVISE":
            # 有明确返工指令且仍有被打回的任务：必须先返工
            return ["researcher"]
        # 这里是真正的选择：继续做下一个任务，或先让 Critic 做一次中期检查
        return ["researcher", "critic"]

    # 全部任务完成
    decision = (critique or {}).get("decision")
    if decision == "PASS":
        return ["writer"]
    if decision == "REPLAN":
        return ["planner"]
    # 无评审，或返工/补题已完成（REVIEW）→ 交给 Critic（重新）评审。
    # 注意：REVISE 但没有 pending 的情况也走这里——说明返工刚做完，
    # Researcher 会把 critique 置为 REVIEW；即便缺失该标记，复审也是正确选择，
    # 绝不能把没有任务可做的 Researcher 反复派出导致空转。
    return ["critic"]


def _rule_based_next(state: ResearchState) -> str:
    """确定性兜底路由。"""
    allowed = _allowed_next(state)
    if "writer" in allowed:
        return "writer"
    if "planner" in allowed:
        return "planner"
    if "researcher" in allowed:
        return "researcher"
    return allowed[0]


def _state_digest(state: ResearchState) -> str:
    tasks = state.get("tasks") or []
    findings = state.get("findings") or {}
    done = sum(1 for t in tasks if t.get("status") == "done")
    lines = [
        f"- 研究问题：{state.get('query', '')}",
        f"- 子任务：{len(tasks)} 个，已完成 {done} 个",
        f"- 已评审轮数：{state.get('iteration', 0)}/{state.get('max_iterations', 4)}",
    ]
    critique = state.get("critique")
    if critique:
        lines.append(f"- 最近评审结论：{critique.get('decision')}"
                     f"（{critique.get('comment', '')[:80]}）")
    return "\n".join(lines)


def decide_next(state: ResearchState) -> tuple[str, str]:
    """
    返回 (下一节点, 决策理由)。
    单一合法目标时不调用 LLM；有多个选项时由 LLM handoff 工具决断，
    非法选择回退到规则路由——既动态又不会失控。
    """
    allowed = _allowed_next(state)
    if len(allowed) == 1:
        return allowed[0], f"唯一合法路径：{allowed[0]}"

    @tool
    def handoff_to(next_agent: str, reason: str) -> str:
        """把控制权移交给下一个智能体。

        参数：
        - next_agent: 下一个智能体名称
        - reason: 一句话中文理由
        """
        return f"handoff → {next_agent}"

    try:
        # 调度决策是轻量任务：关闭深度思考，避免耗时与 token 浪费
        llm = get_llm(temperature=0.0, max_tokens=2048, thinking=False)
        resp = llm.bind_tools([handoff_to]).invoke([
            SystemMessage(content=(
                "你是多智能体研究系统的调度员。根据当前状态，从允许的目标中选择下一个智能体。"
                "可选目标：" + "、".join(allowed) + "。"
                "planner=拆解/补充研究任务；researcher=执行或返工研究；"
                "critic=评审当前研究质量；writer=撰写最终报告（需要评审通过）。"
                "必须且只能调用 handoff_to 工具。"
            )),
            HumanMessage(content=f"当前状态：\n{_state_digest(state)}"),
        ])
        if resp.tool_calls:
            args = resp.tool_calls[0].get("args") or {}
            target = str(args.get("next_agent", ""))
            reason = str(args.get("reason", ""))
            if target in allowed:
                return target, f"LLM 调度：{reason or target}"
        raise ValueError("LLM 未返回合法 handoff")
    except Exception as exc:
        target = _rule_based_next(state)
        return target, f"LLM 调度失败，规则兜底 → {target}（{describe_exc(exc)}）"


def node_supervisor(state: ResearchState) -> Command:
    """Supervisor 节点：读取黑板 → 决定 handoff 目标，不产生业务数据。"""
    nxt, reason = decide_next(state)
    return Command(goto=nxt, update={"trace": [f"Supervisor → {nxt}：{reason}"]})
