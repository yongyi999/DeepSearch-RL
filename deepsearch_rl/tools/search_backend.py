# -*- coding: utf-8 -*-
"""
搜索引擎后端
============

统一封装多家搜索服务，返回相同结构的结果列表。默认按可用 key 自动选择后端。
所有后端返回 ``List[SearchItem]``，再由 search.py 格式化为给模型的文本。

支持后端：
    serper   https://serper.dev   （便宜、稳定，推荐）
    serpapi  https://serpapi.com
    bing     Azure Bing Search v7
    brave    Brave Search API
    tavily   Tavily Search（面向 LLM）
    ddg      DuckDuckGo（免费，无需 key，仅建议调试/兜底）

只用标准库 urllib（避免额外依赖），超时与异常由上层统一分类、重试。
"""

from __future__ import annotations

import abc
import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional

from .exceptions import ToolError, ToolErrorType, classify_http_status

# 浏览器 UA，降低被反爬概率
_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)


@dataclass
class SearchItem:
    title: str
    url: str
    snippet: str

    def to_dict(self) -> dict:
        return {"title": self.title, "url": self.url, "snippet": self.snippet}


def _http_request(
    url: str,
    *,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    payload: Optional[bytes] = None,
    timeout: float = 15.0,
) -> tuple:
    """发起 HTTP 请求，返回 (status_code, text)。把 HTTP 层错误转成 ToolError。"""
    req = urllib.request.Request(url, data=payload, method=method)
    req.add_header("User-Agent", _UA)
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:  # type: ignore[attr-defined]
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise classify_http_status(e.code, body)
    except Exception as e:  # 网络/超时类
        from .exceptions import classify_exception

        raise classify_exception(e)


class SearchProvider(abc.ABC):
    name: str = "base"

    def __init__(self, timeout: float = 15.0) -> None:
        self.timeout = timeout

    @abc.abstractmethod
    def search(self, query: str, key: Optional[str] = None, num: int = 5) -> List[SearchItem]:
        ...


class SerperProvider(SearchProvider):
    name = "serper"

    def search(self, query, key=None, num=5):
        if not key:
            raise ToolError(ToolErrorType.AUTH, "serper 需要 API key")
        body = json.dumps({"q": query, "num": num, "gl": "us", "hl": "en"}).encode()
        status, text = _http_request(
            "https://google.serper.dev/search",
            method="POST",
            headers={"X-API-KEY": key, "Content-Type": "application/json"},
            payload=body,
            timeout=self.timeout,
        )
        data = json.loads(text)
        items: List[SearchItem] = []
        for r in data.get("organic", [])[:num]:
            items.append(
                SearchItem(
                    title=r.get("title", ""),
                    url=r.get("link", ""),
                    snippet=r.get("snippet", ""),
                )
            )
        return items


class SerpApiProvider(SearchProvider):
    name = "serpapi"

    def search(self, query, key=None, num=5):
        if not key:
            raise ToolError(ToolErrorType.AUTH, "serpapi 需要 API key")
        params = urllib.parse.urlencode(
            {"q": query, "api_key": key, "engine": "google", "num": num}
        )
        status, text = _http_request(
            f"https://serpapi.com/search.json?{params}", timeout=self.timeout
        )
        data = json.loads(text)
        if "error" in data:
            raise ToolError(ToolErrorType.QUOTA_EXHAUSTED, str(data["error"]))
        items = []
        for r in data.get("organic_results", [])[:num]:
            items.append(
                SearchItem(
                    title=r.get("title", ""),
                    url=r.get("link", ""),
                    snippet=r.get("snippet", ""),
                )
            )
        return items


class BingProvider(SearchProvider):
    name = "bing"

    def search(self, query, key=None, num=5):
        if not key:
            raise ToolError(ToolErrorType.AUTH, "bing 需要 API key")
        params = urllib.parse.urlencode({"q": query, "count": num})
        status, text = _http_request(
            f"https://api.bing.microsoft.com/v7.0/search?{params}",
            headers={"Ocp-Apim-Subscription-Key": key},
            timeout=self.timeout,
        )
        data = json.loads(text)
        items = []
        for r in data.get("webPages", {}).get("value", [])[:num]:
            items.append(
                SearchItem(
                    title=r.get("name", ""),
                    url=r.get("url", ""),
                    snippet=r.get("snippet", ""),
                )
            )
        return items


class BraveProvider(SearchProvider):
    name = "brave"

    def search(self, query, key=None, num=5):
        if not key:
            raise ToolError(ToolErrorType.AUTH, "brave 需要 API key")
        params = urllib.parse.urlencode({"q": query, "count": num})
        status, text = _http_request(
            f"https://api.search.brave.com/res/v1/web/search?{params}",
            headers={"X-Subscription-Token": key, "Accept": "application/json"},
            timeout=self.timeout,
        )
        data = json.loads(text)
        items = []
        for r in data.get("web", {}).get("results", [])[:num]:
            items.append(
                SearchItem(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    snippet=r.get("description", ""),
                )
            )
        return items


class TavilyProvider(SearchProvider):
    name = "tavily"

    def search(self, query, key=None, num=5):
        if not key:
            raise ToolError(ToolErrorType.AUTH, "tavily 需要 API key")
        body = json.dumps(
            {"query": query, "max_results": num, "search_depth": "basic"}
        ).encode()
        status, text = _http_request(
            "https://api.tavily.com/search",
            method="POST",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            payload=body,
            timeout=self.timeout,
        )
        data = json.loads(text)
        items = []
        for r in data.get("results", [])[:num]:
            items.append(
                SearchItem(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    snippet=r.get("content", ""),
                )
            )
        return items


class DuckDuckGoProvider(SearchProvider):
    """免费兜底：优先使用 ddgs 包；未安装时解析 html.duckduckgo.com。"""

    name = "ddg"

    def search(self, query, key=None, num=5):
        try:
            from ddgs import DDGS  # type: ignore

            items = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=num):
                    items.append(
                        SearchItem(
                            title=r.get("title", ""),
                            url=r.get("href", ""),
                            snippet=r.get("body", ""),
                        )
                    )
            return items
        except ImportError:
            pass
        # HTML 兜底
        params = urllib.parse.urlencode({"q": query})
        status, text = _http_request(
            f"https://html.duckduckgo.com/html/?{params}", timeout=self.timeout
        )
        # 轻量解析结果块
        import re

        items = []
        for block in re.findall(
            r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>'
            r'[\s\S]*?<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
            text,
        ):
            url, title, snippet = block
            url = urllib.parse.unquote(re.sub(r"^//duckduckgo.com/l/\?uddg=", "", url))
            title = re.sub(r"<[^>]+>", "", title)
            snippet = re.sub(r"<[^>]+>", "", snippet)
            items.append(SearchItem(title=title, url=url, snippet=snippet))
            if len(items) >= num:
                break
        return items


PROVIDERS: Dict[str, type] = {
    "serper": SerperProvider,
    "serpapi": SerpApiProvider,
    "bing": BingProvider,
    "brave": BraveProvider,
    "tavily": TavilyProvider,
    "ddg": DuckDuckGoProvider,
}


def build_provider(name: str, timeout: float = 15.0) -> SearchProvider:
    name = (name or "ddg").lower()
    if name not in PROVIDERS:
        raise ToolError(
            ToolErrorType.BAD_REQUEST,
            f"未知搜索后端: {name}，可选 {list(PROVIDERS)}",
        )
    return PROVIDERS[name](timeout=timeout)


__all__ = ["SearchItem", "SearchProvider", "PROVIDERS", "build_provider"]
