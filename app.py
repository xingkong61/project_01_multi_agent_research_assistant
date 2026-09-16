"""
app.py — 多智能体研究助手的 Web 界面（Streamlit）

用途：把 CLI 版包装成一个可让别人在浏览器里使用的网页。输入研究问题 →
实时看到每个智能体的工作轨迹与进度 → 在线查看带引用的 Markdown 报告并下载。

本地运行：
    streamlit run app.py
服务器部署（供他人访问）：
    streamlit run app.py --server.address 0.0.0.0 --server.port 8501
    （生产环境建议前置 Nginx + HTTPS，见 README「Web 部署」小节）

复用 main.stream_research() 这一流式核心，与 CLI 共享同一套图逻辑；
model 通过 contextvar per-request 注入，多人并发不互相污染。
"""
from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# 确保以脚本方式运行时能导入同目录模块
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import llm
from main import preflight_check, slugify, stream_research

st.set_page_config(page_title="多智能体研究助手", page_icon="🔬", layout="wide")


# ------------------------------------------------------------------ #
# 侧边栏：配置
# ------------------------------------------------------------------ #
with st.sidebar:
    st.header("⚙️ 设置")

    model = st.text_input(
        "模型标识",
        value=os.getenv("LLM_MODEL", llm.DEFAULT_MODEL),
        help="如 qwen3.8-flash、groq/llama-3.3-70b-versatile、openai/gpt-4o-mini",
    )
    max_iter = st.slider("评审-返工最大轮数", 1, 6, 4,
                         help="越大越严谨但更慢、更耗额度")
    save_reports = st.checkbox("同时保存报告到服务器 reports/", value=False)

    st.divider()
    st.caption("默认走阿里云百炼兼容端点，需在 .env 配好 LLM_API_KEY。"
               "Tavily 可选，不填仍可用 arXiv + 网页抓取。")


# ------------------------------------------------------------------ #
# 主区：输入
# ------------------------------------------------------------------ #
st.title("🔬 多智能体研究助手")
st.caption("Supervisor 动态调度 · Planner / Researcher / Critic / Writer · 生成带引用的研究报告")

query = st.text_area(
    "研究问题",
    placeholder="例如：RAG 检索增强生成在 2025 年有哪些重要进展？",
    height=90,
)
run_btn = st.button("开始研究", type="primary", use_container_width=True)


def _render_report(state: dict):
    """渲染最终报告 + 引用统计 + 下载按钮。"""
    report = state.get("report_markdown", "")
    findings = state.get("findings", {})
    tasks = state.get("tasks", [])
    errors = state.get("errors", [])
    iterations = state.get("iteration", 0)

    n_sources = sum(len(f.get("sources", [])) for f in findings.values())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("子任务", len(tasks))
    c2.metric("评审轮数", iterations)
    c3.metric("引用来源", n_sources)
    c4.metric("异常", len(errors))

    if errors:
        with st.expander(f"⚠ {len(errors)} 条运行提示", expanded=False):
            for e in errors:
                st.text(f"- {e}")

    st.markdown("---")
    st.markdown(report or "_（未生成报告）_")

    st.download_button(
        "⬇️ 下载 Markdown 报告",
        data=(report or "").encode("utf-8"),
        file_name=f"{slugify(query)}.md",
        mime="text/markdown",
    )
    if save_reports and report:
        os.makedirs("reports", exist_ok=True)
        from datetime import datetime
        path = os.path.join("reports", f"{datetime.now():%Y%m%d-%H%M%S}-{slugify(query)}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(report)
        st.caption(f"已保存到服务器：{path}")


# ------------------------------------------------------------------ #
# 执行：流式进度
# ------------------------------------------------------------------ #
if run_btn:
    if not query.strip():
        st.warning("请先输入研究问题。")
        st.stop()

    # 启动前预检：LLM 配置缺失直接明确报错，避免跑到一半才失败
    problems = preflight_check(model or None, quiet=True)
    hard = [p for p in problems if not p.startswith("[提示]")]
    hints = [p for p in problems if p.startswith("[提示]")]
    for h in hints:
        st.info(h.replace("[提示] ", ""))
    if hard:
        st.error("配置不完整，无法启动：\n\n" + "\n".join(f"- {p}" for p in hard))
        st.stop()

    progress = st.progress(0.0, text="准备中…")
    trace_box = st.empty()
    traces: list[str] = []
    final_state: dict = {}

    try:
        for kind, payload in stream_research(query.strip(), model or None, max_iter):
            if kind == "trace":
                traces.append(payload)
                trace_box.code("\n".join(traces[-15:]), language=None)
            elif kind == "status":
                total = payload["tasks_total"] or 1
                done = payload["tasks_done"]
                frac = min(0.95, (done / total) * 0.9)
                progress.progress(max(0.05, frac),
                                  text=f"已完成任务 {done}/{total} · 已评审 {payload['iteration']} 轮")
            elif kind == "done":
                final_state = payload
    except Exception as exc:  # noqa: BLE001 — Web 层需兜住一切异常给出友好提示
        progress.empty()
        trace_box.empty()
        st.error(f"运行出错：{exc}")
        st.stop()

    progress.progress(1.0, text="完成")
    st.success("研究完成！")
    _render_report(final_state)
else:
    st.info("👈 输入一个研究问题，点击「开始研究」。一次完整研究通常需要几分钟，请耐心等待。")
