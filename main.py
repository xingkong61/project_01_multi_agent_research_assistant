"""
main.py — 命令行入口

用法：
    python main.py "RAG 检索增强生成 2025 年有哪些重要进展？"
    python main.py --max-iterations 3 --model qwen3.8-flash
    python main.py --output reports/rag.md "你的问题"
    python main.py                  # 不带参数则进入交互模式

执行过程中实时打印每个智能体的工作轨迹（trace），报告默认保存到 reports/ 目录。
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime

from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph import build_graph, compute_recursion_limit
from langgraph.types import Overwrite

# trace / errors / critique_history 在状态中是 operator.add reducer，
# 从 stream 增量里合并时需追加；其余字段直接覆盖。
_APPEND_KEYS = {"trace", "errors", "critique_history"}


def _merge_update(latest: dict, update: dict) -> None:
    """把单个节点的 stream 增量合并进累计状态。

    add-reducer 字段默认追加；但当节点用 Overwrite 显式重置（如 Planner 初始规划
    清空 critique_history）时，必须整体替换而非追加——否则会像 [] 那样语义丢失。
    """
    for k, v in update.items():
        if isinstance(v, Overwrite):
            latest[k] = list(v.value)
        elif k in _APPEND_KEYS and isinstance(v, list):
            latest.setdefault(k, []).extend(v)
        else:
            latest[k] = v

BANNER = r"""
  ___  ___ ___  ___  _   _ _____
 / __|| __/ _ \|   \| | | / __|   研究智能体 v2.0
 \__ \| _| (_) | |) | |_| \__ \   Planner · Researcher · Critic · Writer
 |___/|___\___/|___/ \___/|___/   Supervisor 动态调度 · ReAct 工具调用
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="多智能体研究助手：输入问题，输出带引用的 Markdown 研究报告",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("query", nargs="?", help="研究问题（省略则交互输入）")
    parser.add_argument("--model", default=None,
                        help="模型标识，如 qwen3.8-flash、groq/llama-3.3-70b-versatile、"
                             "openai/gpt-4o、deepseek/deepseek-chat，默认读 LLM_MODEL 环境变量")
    parser.add_argument("--max-iterations", type=int, default=4,
                        help="评审-返工最大轮数（默认 4）")
    parser.add_argument("--output", "-o", default=None,
                        help="报告保存路径；不给则自动存入 reports/ 目录")
    parser.add_argument("--no-save", action="store_true",
                        help="只在屏幕打印，不落盘报告")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="不打印执行轨迹，只输出报告")
    return parser.parse_args(argv)


def slugify(text: str, max_len: int = 40) -> str:
    text = re.sub(r"[\\/:*?\"<>|\s]+", "-", text.strip()).strip("-")
    return (text[:max_len].rstrip("-") or "report")


# 明显是示例/占位符的 Key 片段（.env.example 里的样子），命中即视为未真正配置。
_PLACEHOLDER_HINTS = ("xxxx", "your-", "your_", "sk-...", "...", "changeme", "placeholder")


def _looks_placeholder(value: str) -> bool:
    v = value.strip().lower()
    if not v:
        return True
    return any(h in v for h in _PLACEHOLDER_HINTS)


def preflight_check(model: str | None, quiet: bool) -> list[str]:
    """启动前校验 LLM / 搜索配置，返回人类可读的问题列表（空表示通过）。

    - LLM：按 llm.get_llm 的解析规则确定需要哪个 Key，缺失或仍是占位符 → 报错；
      这一步能在真正调用模型前给出明确提示，而不是跑完整个流程才笼统失败。
    - Tavily：可选，未配置只降级为"无联网搜索"，给一条提示而非阻断。
    """
    import llm as llm_mod

    problems: list[str] = []
    spec = (model or os.getenv("LLM_MODEL") or llm_mod.DEFAULT_MODEL).strip()
    if "/" in spec:
        provider, _ = spec.split("/", 1)
    else:
        provider = "custom" if os.getenv("LLM_BASE_URL") else "openai"

    if provider == "custom":
        base_url = os.getenv("LLM_BASE_URL", "")
        key = os.getenv("LLM_API_KEY", "")
        if not base_url or _looks_placeholder(base_url):
            problems.append(f"LLM_BASE_URL 未配置或仍是占位符（当前供应商 {provider}）")
        if _looks_placeholder(key):
            problems.append("LLM_API_KEY 未配置或仍是占位符（请在 .env 填入真实密钥）")
    elif provider in llm_mod._PROVIDERS:
        _, key_env = llm_mod._PROVIDERS[provider]
        if _looks_placeholder(os.getenv(key_env, "")):
            problems.append(f"{key_env} 未配置或仍是占位符（供应商 {provider}）")
    else:
        supported = ", ".join(sorted(llm_mod._PROVIDERS) + ["custom"])
        problems.append(f"不支持的模型供应商：{provider}（支持：{supported}）")

    if _looks_placeholder(os.getenv("TAVILY_API_KEY", "")):
        problems.append("[提示] 未配置有效 TAVILY_API_KEY：联网搜索不可用，"
                        "Researcher 仍可用 arXiv 检索与网页抓取完成研究")
    return problems


def stream_research(query: str, model: str | None = None,
                    max_iterations: int = 4):
    """研究流程的流式核心（CLI 与 Web 共用）。

    逐步 yield：
      - ("trace", <str>)   每产生一条人类可读轨迹时
      - ("status", <dict>) 节点写入的关键状态快照（任务/评审进度）
    结束时 yield ("done", <final_state_dict>)。

    不 print、不改写进程级环境变量：model 以参数形式透传给图，保证并发安全
    （多人同时用不会互相覆盖模型选择）。
    """
    app = build_graph()
    initial = {"query": query, "max_iterations": max_iterations}
    config: dict = {"recursion_limit": compute_recursion_limit(max_iterations)}
    latest: dict = {}

    # model 以 contextvar 承载（并发隔离），而非改写进程级 os.environ
    import llm as _llm
    token = _llm.set_request_model(model) if model else None
    try:
        for chunk in app.stream(initial, config=config, stream_mode="updates"):
            for _node, update in chunk.items():
                if not isinstance(update, dict):
                    continue
                _merge_update(latest, update)
                for line in update.get("trace", []):
                    yield "trace", line
                yield "status", {
                    "iteration": latest.get("iteration", 0),
                    "tasks_done": sum(1 for t in latest.get("tasks", [])
                                      if t.get("status") == "done"),
                    "tasks_total": len(latest.get("tasks", [])),
                }
    finally:
        if token is not None:
            _llm._request_model.reset(token)
    yield "done", latest


def run(query: str, model: str | None, max_iterations: int,
        quiet: bool) -> dict:
    """执行研究流程，返回最终状态（同时打印进度）。"""
    if not quiet:
        print(f"\n研究问题：{query}\n" + "-" * 64)

    latest: dict = {}
    for kind, payload in stream_research(query, model, max_iterations):
        if kind == "trace" and not quiet:
            print(f"  • {payload}")
        elif kind == "done":
            latest = payload
    return latest


def save_report(query: str, report: str, output: str | None) -> str:
    """把报告落盘，返回文件路径。"""
    if output:
        path = output
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    else:
        os.makedirs("reports", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = os.path.join("reports", f"{ts}-{slugify(query)}.md")

    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    return path


def main(argv: list[str] | None = None) -> int:
    # 中文 Windows 控制台默认 GBK，输出 •/⚠ 等符号会 UnicodeEncodeError 崩溃；
    # 尽力把 stdout/stderr 切到 UTF-8 并对无法编码字符降级替换（Python 3.7+）。
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    load_dotenv()
    args = parse_args(argv)

    query = args.query
    if not query:
        try:
            query = input("请输入研究问题：").strip()
        except EOFError:
            query = ""
    if not query:
        print("未提供研究问题，用法：python main.py \"你的问题\"")
        return 2

    if not args.quiet:
        print(BANNER)

    # 启动前配置预检：LLM 不可用直接快速失败并给出明确原因；
    # Tavily 缺失只是可选能力降级，打印提示后继续。
    problems = preflight_check(args.model, args.quiet)
    hard_errors = [p for p in problems if not p.startswith("[提示]")]
    hints = [p for p in problems if p.startswith("[提示]")]
    for h in hints:
        print(h.replace("[提示] ", "*  "))
    if hard_errors:
        print("\n[配置错误] 无法启动，请修正以下项后重试（参考 .env.example）：")
        for p in hard_errors:
            print(f"  - {p}")
        return 2

    try:
        state = run(query, args.model, args.max_iterations, args.quiet)
    except KeyboardInterrupt:
        print("\n已中断。")
        return 130

    report = state.get("report_markdown", "")
    errors = state.get("errors", [])
    tasks = state.get("tasks", [])
    findings = state.get("findings", {})
    iterations = state.get("iteration", 0)

    if not report:
        print("\n[错误] 未生成报告，请检查 API Key 与网络（可设置 LLM_MODEL 切换模型）。")
        for e in errors:
            print(f"  - {e}")
        return 1

    if not args.quiet:
        print("-" * 64)
    print(f"\n{report}\n")

    if not args.quiet:
        n_sources = sum(len(f.get("sources", [])) for f in findings.values())
        print("-" * 64)
        print(f"子任务 {len(tasks)} 个 | 评审 {iterations} 轮 | "
              f"引用来源 {n_sources} 条 | 异常 {len(errors)} 个")
        for e in errors:
            print(f"  ⚠ {e}")

    if not args.no_save:
        path = save_report(query, report, args.output)
        print(f"\n报告已保存：{os.path.abspath(path)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
