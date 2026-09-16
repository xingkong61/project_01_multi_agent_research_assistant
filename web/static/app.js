// 研究文牍 · 前端逻辑（无框架，配合 /api/research 的 SSE 流）
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const els = {
    query: $("query"), model: $("model"), iters: $("iters"),
    submit: $("submit"), progress: $("progress"), traceLog: $("traceLog"),
    statusLine: $("statusLine"), report: $("report"), reportBody: $("reportBody"),
    reportMeta: $("reportMeta"), download: $("download"),
  };

  // 配置 marked（若 CDN 不可用则降级为纯文本）
  if (window.marked) {
    window.marked.setOptions({ breaks: true, gfm: true });
  }

  function addTrace(line) {
    const li = document.createElement("li");
    li.textContent = line;
    els.traceLog.appendChild(li);
    li.scrollIntoView({ block: "nearest" });
  }

  function showError(msg) {
    let box = document.querySelector(".error-note");
    if (!box) {
      box = document.createElement("div");
      box.className = "error-note";
      els.progress.classList.remove("hidden");
      els.progress.parentNode.insertBefore(box, els.progress.nextSibling);
    }
    box.textContent = "⚠ " + msg;
  }

  function clearError() {
    const box = document.querySelector(".error-note");
    if (box) box.remove();
  }

  function renderReport(md) {
    if (window.marked) {
      els.reportBody.innerHTML = window.marked.parse(md || "");
    } else {
      const pre = document.createElement("pre");
      pre.style.whiteSpace = "pre-wrap";
      pre.textContent = md || "";
      els.reportBody.replaceChildren(pre);
    }
  }

  async function runResearch() {
    const query = els.query.value.trim();
    if (query.length < 2) { showError("请输入至少两个字符的研究议题。"); return; }

    clearError();
    els.submit.disabled = true;
    els.submit.textContent = "编纂中……";
    els.traceLog.replaceChildren();
    els.report.classList.add("hidden");
    els.progress.classList.remove("hidden");
    els.statusLine.textContent = "正在拆解议题、检索资料……";

    try {
      const resp = await fetch("/api/research", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        // 凭据由浏览器 Basic Auth 弹窗/会话提供；同源请求自动携带
        credentials: "same-origin",
        body: JSON.stringify({
          query,
          model: els.model.value.trim() || null,
          max_iterations: parseInt(els.iters.value, 10),
        }),
      });

      if (resp.status === 401) {
        showError("需要访问口令（401）。请刷新页面并按提示输入用户名与口令。");
        finish(); return;
      }
      if (resp.status === 429) {
        showError((await resp.json()).detail || "请求过于频繁。");
        finish(); return;
      }
      if (!resp.ok) {
        let d = "服务端错误";
        try { d = (await resp.json()).detail || d; } catch (_) {}
        showError(d + `（HTTP ${resp.status}）`);
        finish(); return;
      }

      // 读取 SSE 流
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let sep;
        // SSE 事件以空行分隔
        while ((sep = buffer.indexOf("\n\n")) !== -1) {
          const rawEvent = buffer.slice(0, sep);
          buffer = buffer.slice(sep + 2);
          handleEvent(rawEvent);
        }
      }
    } catch (err) {
      showError("连接中断：" + err.message);
    } finally {
      finish();
    }
  }

  function handleEvent(raw) {
    let event = "message", dataStr = "";
    for (const line of raw.split("\n")) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) dataStr += line.slice(5).trim();
    }
    if (!dataStr) return;
    let data;
    try { data = JSON.parse(dataStr); } catch (_) { return; }

    if (event === "trace") {
      addTrace(data.line);
    } else if (event === "status") {
      const total = data.tasks_total || 0;
      els.statusLine.textContent =
        `已完成任务 ${data.tasks_done}/${total} · 已评审 ${data.iteration} 轮`;
    } else if (event === "result") {
      renderReport(data.report);
      els.reportMeta.textContent =
        `子任务 ${data.tasks} 个 · 评审 ${data.iterations} 轮 · 引用来源 ${data.sources} 条`;
      const blob = new Blob([data.report || ""], { type: "text/markdown;charset=utf-8" });
      els.download.href = URL.createObjectURL(blob);
      els.report.classList.remove("hidden");
      if (data.errors && data.errors.length) {
        // 运行提示不阻断展示，仅在纪要末尾附注
        addTrace(`（系统提示：${data.errors.length} 条运行留痕，详见报告脚注区）`);
      }
      els.statusLine.textContent = "研究完成 ✓";
    } else if (event === "error") {
      showError(data.detail || "研究过程中出现错误。");
    }
  }

  function finish() {
    els.submit.disabled = false;
    els.submit.textContent = "开始研究";
  }

  els.submit.addEventListener("click", runResearch);
  els.query.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") runResearch();
  });
})();
