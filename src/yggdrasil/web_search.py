"""
网络搜索工具 - 赋予 Agent 搜索互联网的能力
使用 DuckDuckGo 搜索 (无需 API Key) + httpx 抓取网页内容
"""

import json
import logging
import re
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Any

logger = logging.getLogger(__name__)

# DuckDuckGo HTML 搜索 (无需第三方库)
DDGS_URL = "https://html.duckduckgo.com/html/"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)
REQUEST_TIMEOUT = 15  # seconds


class _DDGResultParser(HTMLParser):
    """解析 DuckDuckGo HTML 搜索结果页面"""

    def __init__(self):
        super().__init__()
        self.results: list[dict] = []
        self._in_result = False
        self._in_title = False
        self._in_snippet = False
        self._current: dict = {}
        self._text_buf: list[str] = []

    def handle_starttag(self, tag, attrs):
        attr_dict = dict(attrs)
        cls = attr_dict.get("class", "")

        if tag == "div" and "result__body" in cls:
            self._in_result = True
            self._current = {"title": "", "url": "", "snippet": ""}
        elif self._in_result and tag == "a" and "result__a" in cls:
            self._in_title = True
            self._text_buf = []
            href = attr_dict.get("href", "")
            # DuckDuckGo 会用 redirect URL，提取真实 URL
            if "uddg=" in href:
                try:
                    real = urllib.parse.parse_qs(urllib.parse.urlparse(href).query).get("uddg", [""])[0]
                    href = real
                except Exception:
                    pass
            self._current["url"] = href
        elif self._in_result and tag == "a" and "result__snippet" in cls:
            self._in_snippet = True
            self._text_buf = []

    def handle_endtag(self, tag):
        if self._in_title and tag == "a":
            self._in_title = False
            self._current["title"] = "".join(self._text_buf).strip()
        elif self._in_snippet and tag == "a":
            self._in_snippet = False
            self._current["snippet"] = "".join(self._text_buf).strip()
        elif self._in_result and tag == "div":
            if self._current.get("title"):
                self.results.append(self._current)
            self._in_result = False
            self._current = {}

    def handle_data(self, data):
        if self._in_title or self._in_snippet:
            self._text_buf.append(data)


def web_search(query: str, max_results: int = 8) -> dict:
    """
    使用 DuckDuckGo 搜索互联网。

    参数:
        query: 搜索关键词
        max_results: 最大返回结果数 (默认8)

    返回:
        {"query": str, "results": [{"title", "url", "snippet"}]}
    """
    try:
        data = urllib.parse.urlencode({"q": query, "kl": ""}).encode("utf-8")
        req = urllib.request.Request(
            DDGS_URL,
            data=data,
            headers={"User-Agent": USER_AGENT},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            html = resp.read().decode("utf-8", errors="replace")

        parser = _DDGResultParser()
        parser.feed(html)
        results = parser.results[:max_results]

        return {
            "query": query,
            "total_results": len(results),
            "results": results,
        }
    except Exception as e:
        logger.error(f"Web search failed: {e}", exc_info=True)
        return {"query": query, "total_results": 0, "results": [], "error": str(e)}


class _TextExtractor(HTMLParser):
    """从 HTML 中提取纯文本"""

    def __init__(self):
        super().__init__()
        self.pieces: list[str] = []
        self._skip_tags = {"script", "style", "noscript", "nav", "header", "footer", "aside"}
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._skip_tags:
            self._skip_depth += 1
        if tag in ("p", "br", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr"):
            self.pieces.append("\n")

    def handle_endtag(self, tag):
        if tag in self._skip_tags:
            self._skip_depth = max(0, self._skip_depth - 1)

    def handle_data(self, data):
        if self._skip_depth == 0:
            text = data.strip()
            if text:
                self.pieces.append(text + " ")

    def get_text(self) -> str:
        raw = "".join(self.pieces)
        # 合并多余空行
        return re.sub(r"\n{3,}", "\n\n", raw).strip()


def fetch_webpage(url: str, max_chars: int = 8000) -> dict:
    """
    抓取网页并提取正文文本。

    参数:
        url: 网页 URL
        max_chars: 最大返回字符数 (默认8000)

    返回:
        {"url": str, "title": str, "content": str, "length": int}
    """
    # 安全校验：只允许 http/https
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return {"url": url, "error": f"不支持的协议: {parsed.scheme}，只允许 http/https"}

    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            },
        )
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" not in content_type and "application/xhtml" not in content_type:
                return {"url": url, "error": f"不是 HTML 页面: {content_type}"}

            html = resp.read(512 * 1024).decode("utf-8", errors="replace")  # 最多读 512KB

        # 提取标题
        title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        title = title_match.group(1).strip() if title_match else ""

        # 提取正文
        extractor = _TextExtractor()
        extractor.feed(html)
        text = extractor.get_text()

        # 截断
        truncated = len(text) > max_chars
        text = text[:max_chars]

        return {
            "url": url,
            "title": title,
            "content": text,
            "length": len(text),
            "truncated": truncated,
        }
    except Exception as e:
        logger.error(f"Fetch webpage failed ({url}): {e}", exc_info=True)
        return {"url": url, "error": str(e)}
