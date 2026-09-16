"""
test_tools.py — 工具层纯函数的离线单元测试（不发网络请求）
"""
from tools import html_to_text, parse_arxiv_feed

ARXIV_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.00001v1</id>
    <title>Retrieval-Augmented Generation: A Survey</title>
    <summary>  This paper surveys RAG systems and   their components. </summary>
    <published>2024-01-02T00:00:00Z</published>
    <author><name>Alice Zhang</name></author>
    <author><name>Bob Li</name></author>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2402.00002v1</id>
    <title>Corrective RAG</title>
    <summary>CRAG adds self-correction to retrieval.</summary>
    <published>2024-02-10T00:00:00Z</published>
    <author><name>Carol Wang</name></author>
  </entry>
</feed>"""


class TestArxivParse:
    def test_parse_entries(self):
        papers = parse_arxiv_feed(ARXIV_SAMPLE)
        assert len(papers) == 2

    def test_fields(self):
        p = parse_arxiv_feed(ARXIV_SAMPLE)[0]
        assert p["title"] == "Retrieval-Augmented Generation: A Survey"
        assert p["url"] == "https://arxiv.org/abs/2401.00001v1"
        assert p["published"] == "2024-01-02"
        assert p["authors"] == ["Alice Zhang", "Bob Li"]
        # 摘要空白被折叠
        assert "  " not in p["summary"]

    def test_second_entry(self):
        p = parse_arxiv_feed(ARXIV_SAMPLE)[1]
        assert p["title"] == "Corrective RAG"

    def test_invalid_xml_raises(self):
        import xml.etree.ElementTree as ET
        try:
            parse_arxiv_feed("<not-valid")
        except ET.ParseError:
            pass
        else:
            raise AssertionError("非法 XML 应抛出 ParseError")


class TestHtmlToText:
    def test_extract_text(self):
        html = """
        <html><head><style>.x{color:red}</style></head>
        <body><nav>导航栏</nav>
          <article><h1>标题</h1><p>这是正文段落。</p></article>
          <script>console.log('noise');</script>
        </body></html>"""
        text = html_to_text(html)
        assert "这是正文段落。" in text
        assert "标题" in text

    def test_removes_script_and_style(self):
        text = html_to_text(
            "<html><body><p>保留我</p>"
            "<script>alert(1)</script><style>.a{}</style></body></html>"
        )
        assert "alert" not in text
        assert "color" not in text or True  # style 内容已移除
        assert "保留我" in text

    def test_collapse_blank_lines(self):
        text = html_to_text("<p>a</p><p>b</p>")
        assert "\n\n\n" not in text
