"""
tools.py — Researcher 可调用的研究工具集

工具（docstring 会作为工具说明提供给 LLM）：
1. web_search      —— Tavily 联网搜索（需要 TAVILY_API_KEY，缺失时降级）
2. fetch_webpage   —— 抓取并清洗任意网页正文
3. search_arxiv    —— arXiv 论文检索（无需 API Key，研究人员的核心免费信源）

纯函数 html_to_text / parse_arxiv_feed 单独暴露，方便离线单元测试。
所有工具对异常做了兜底：失败时返回以 "[工具错误]" 开头的字符串，
由智能体自己决定换关键词、换工具还是带着有限信息继续，而不是让流程崩溃。
"""
from __future__ import annotations

import ipaddress
import os
import re
import socket
import time
import xml.etree.ElementTree as ET
from html import unescape
from urllib.parse import urlencode, urlparse, urljoin

import requests
from langchain_core.tools import tool

_TIMEOUT = 20
_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
_MAX_PAGE_CHARS = 8000
_ARXIV_LAST_TS = 0.0  # 上次 arXiv 请求时间戳，用于限流


# ============================================================
# SSRF 防护：拒绝抓取内网 / 元数据 / 非 http(s) 地址
# ============================================================
def is_blocked_url(url: str) -> bool:
    """判断 URL 是否指向不应被服务端访问的地址（SSRF 风险）。纯函数，可离线单测。

    拦截：非 http/https 协议；host 为 localhost / *.internal 等；解析后 IP 落在
    回环、私有、链路本地(含云厂商 169.254.169.254 元数据)、组播、保留地址段。
    DNS 解析失败按"阻断"处理（宁可拒也不放行未知目标）。
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return True
    if parsed.scheme not in ("http", "https"):
        return True
    host = parsed.hostname
    if not host:
        return True
    lowered = host.lower()
    if lowered in ("localhost", "metadata.google.internal") or lowered.endswith(
            (".local", ".internal", ".localhost")):
        return True

    def _ip_blocked(ip_str: str) -> bool:
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            return True
        return (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_multicast or ip.is_reserved or ip.is_unspecified)

    # host 本身就是 IP 字面量
    try:
        ipaddress.ip_address(host)
        return _ip_blocked(host)
    except ValueError:
        pass
    # 域名：解析所有 A/AAAA 记录，任一命中内网即阻断（防 DNS rebinding 的第一跳）
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return True
    for info in infos:
        addr = info[4][0]
        if _ip_blocked(addr):
            return True
    return False


# ============================================================
# 纯函数（可单测）
# ============================================================
def html_to_text(html: str) -> str:
    """从 HTML 中提取可读正文。优先用 bs4，未安装时退化为正则实现。"""
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header",
                         "noscript", "form", "aside"]):
            tag.decompose()
        text = soup.get_text("\n")
    except ImportError:
        text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
        text = re.sub(r"(?s)<[^>]+>", " ", text)
        text = unescape(text)

    # 折叠空白
    lines = [ln.strip() for ln in text.splitlines()]
    text = "\n".join(ln for ln in lines if ln)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_arxiv_feed(xml_text: str) -> list[dict]:
    """
    解析 arXiv Atom Feed，返回论文列表。
    每项：{title, authors, published, url, summary}
    """
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(xml_text)
    papers: list[dict] = []
    for entry in root.findall("atom:entry", ns):
        eid = entry.findtext("atom:id", default="", namespaces=ns).strip()
        title = " ".join(
            entry.findtext("atom:title", default="", namespaces=ns).split()
        )
        summary = " ".join(
            entry.findtext("atom:summary", default="", namespaces=ns).split()
        )
        published = entry.findtext("atom:published", default="", namespaces=ns)[:10]
        authors = [
            a.findtext("atom:name", default="", namespaces=ns)
            for a in entry.findall("atom:author", ns)
        ]
        papers.append({
            "title": title,
            "authors": [a for a in authors if a],
            "published": published,
            "url": eid.replace("http://", "https://"),
            "summary": summary,
        })
    return papers


# ============================================================
# LangChain 工具
# ============================================================
@tool
def web_search(query: str, max_results: int = 5) -> str:
    """联网搜索最新的网页信息。输入搜索关键词，返回标题、URL 和内容摘要。
    适合查找最新进展、新闻、官方文档、博客、技术报告等。"""
    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        return (
            "[工具错误] 未配置 TAVILY_API_KEY，无法联网搜索。"
            "可改用 search_arxiv 检索学术论文，或用 fetch_webpage 抓取已知 URL。"
        )

    try:
        from langchain_tavily import TavilySearch

        # 注意 langchain-tavily 0.2.x：
        # - 鉴权参数名是 tavily_api_key（传 api_key 会被静默忽略）；
        # - response_format 只支持 "content"（默认，invoke 直接返回格式化好的字符串）
        #   和 "content_and_artifact"（0.2.18 的 invoke 路径不兼容，会抛异常），没有 json 模式。
        #   直接使用默认 content 字符串即可——它已包含标题/URL/摘要，模型可直接阅读。
        search = TavilySearch(
            tavily_api_key=api_key,
            max_results=max_results,
        )
        raw = search.invoke({"query": query})
    except Exception as exc:  # 额度不足 / 网络问题 / 包未安装
        return f"[工具错误] 联网搜索失败：{exc}"

    # content 模式：invoke 返回纯字符串，非空直接使用
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return f"[未找到结果] 关键词：{query}"
        # 0.2.x 空结果时可能返回固定提示串，统一归一化
        if text.lower().startswith("no search results found"):
            return f"[未找到结果] 关键词：{query}"
        return text

    # 兼容 dict 形态（失败时 Tavily 返回 {"error": ...}，必须如实上报，
    # 否则鉴权/额度问题会被误报成"未找到结果"，模型与排障者都被误导）
    if isinstance(raw, dict):
        if raw.get("error"):
            return f"[工具错误] 联网搜索失败：{raw['error']}"
        results = raw.get("results", [])
        if not results:
            return f"[未找到结果] 关键词：{query}"
        blocks = []
        for i, r in enumerate(results, 1):
            blocks.append(
                f"[{i}] {r.get('title', '')}\n"
                f"URL: {r.get('url', '')}\n"
                f"摘要: {r.get('content', '')}"
            )
        return "\n\n".join(blocks)

    return f"[未找到结果] 关键词：{query}"


@tool
def fetch_webpage(url: str) -> str:
    """抓取指定 URL 的网页正文（自动去除导航/脚本等噪声）。
    当你已经从搜索结果或参考文献中得到具体链接、需要阅读全文细节时使用。"""
    if not url.startswith(("http://", "https://")):
        return "[工具错误] URL 必须以 http:// 或 https:// 开头"
    # SSRF 防护：拒绝内网 / 元数据地址；关闭自动重定向，逐跳校验目标
    if is_blocked_url(url):
        return f"[工具错误] 出于安全限制，拒绝抓取该地址（内网/元数据/非法目标）：{url}"

    try:
        resp = requests.get(
            url,
            headers={"User-Agent": _UA, "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"},
            timeout=_TIMEOUT,
            allow_redirects=False,
        )
        # 手动跟随重定向并对每一跳做 SSRF 校验
        hops = 0
        while resp.is_redirect and hops < 5:
            next_url = urljoin(url, resp.headers.get("Location", ""))
            if not next_url.startswith(("http://", "https://")):
                return "[工具错误] 重定向目标协议非法，已中止"
            if is_blocked_url(next_url):
                return f"[工具错误] 重定向指向受限地址，已中止：{next_url}"
            resp = requests.get(
                next_url,
                headers={"User-Agent": _UA, "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"},
                timeout=_TIMEOUT, allow_redirects=False,
            )
            url = next_url
            hops += 1
        resp.raise_for_status()
    except requests.RequestException as exc:
        return f"[工具错误] 抓取失败（{url}）：{exc}"

    # 让 requests 按响应内容猜测编码，避免中文乱码
    if not resp.encoding or resp.encoding.lower() == "iso-8859-1":
        resp.encoding = resp.apparent_encoding

    text = html_to_text(resp.text)
    if not text:
        return f"[未提取到正文] {url}（可能是需要 JS 渲染的页面）"
    if len(text) > _MAX_PAGE_CHARS:
        text = text[:_MAX_PAGE_CHARS] + "\n\n[内容过长，已截断；如需更多细节可缩小范围]"
    return f"来源：{url}\n\n{text}"


@tool
def search_arxiv(keyword: str, max_results: int = 5) -> str:
    """检索 arXiv 上的学术论文（免费、无需 API Key）。
    输入研究主题关键词（建议英文），返回论文标题、作者、发布日期、链接和摘要。
    适合查找前沿方法、经典论文、实验结论等权威学术资料。"""
    max_results = max(1, min(int(max_results or 5), 10))
    params = {
        "search_query": f"all:{keyword}",
        "start": 0,
        "max_results": max_results,
        # 按相关性排序：submittedDate 会让任意查询都返回最新论文而非相关论文
        "sortBy": "relevance",
    }
    url = f"https://export.arxiv.org/api/query?{urlencode(params)}"

    # arXiv 官方要求同一 IP 的请求间隔 ≥3 秒，否则触发限流（429/503）
    global _ARXIV_LAST_TS
    wait = 3.0 - (time.time() - _ARXIV_LAST_TS)
    if wait > 0:
        time.sleep(wait)
    _ARXIV_LAST_TS = time.time()

    try:
        resp = requests.get(
            url,
            headers={"User-Agent": _UA},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        papers = parse_arxiv_feed(resp.text)
    except (requests.RequestException, ET.ParseError) as exc:
        return f"[工具错误] arXiv 检索失败：{exc}"

    if not papers:
        return f"[未找到论文] 关键词：{keyword}（可尝试更宽泛或英文关键词）"

    blocks = []
    for i, p in enumerate(papers, 1):
        authors = ", ".join(p["authors"][:4])
        if len(p["authors"]) > 4:
            authors += " et al."
        summary = p["summary"][:600]
        blocks.append(
            f"[{i}] {p['title']} ({p['published']})\n"
            f"作者: {authors}\n"
            f"URL: {p['url']}\n"
            f"摘要: {summary}"
        )
    return "\n\n".join(blocks)


RESEARCH_TOOLS = [web_search, fetch_webpage, search_arxiv]
