# -*- coding: utf-8 -*-
"""
Open 工具
=========

打开搜索结果中的 URL，抽取网页正文文本。优先使用 trafilatura（抽取质量好，pip 可装），
未安装时回退到 BeautifulSoup，再回退到标准库正则去标签。

包含失败重试与异常分类，并对非 http(s)、二进制/PDF 等做明确反馈。
"""

from __future__ import annotations

import re
import time
from typing import Optional

from ..protocol import OPEN_TAG
from .base import BaseTool
from .cache import PersistentToolCache
from .exceptions import ToolError, ToolErrorType
from .search_backend import _http_request

_MAX_PAGE_CHARS = 6000


class OpenTool(BaseTool):
    """打开网页并返回正文。"""

    name = OPEN_TAG

    def __init__(
        self,
        cache: Optional[PersistentToolCache] = None,
        *,
        max_retries: int = 3,
        timeout: float = 20.0,
        backoff_base: float = 1.5,
        max_chars: int = _MAX_PAGE_CHARS,
    ) -> None:
        super().__init__(cache=cache)
        self.max_retries = max_retries
        self.timeout = timeout
        self.backoff_base = backoff_base
        self.max_chars = max_chars

    # ------------------------------------------------------------------
    def _run(self, argument: str) -> str:
        url = argument.strip()
        if not url:
            raise ToolError(ToolErrorType.BAD_REQUEST, "open 的 URL 为空")
        if not url.startswith(("http://", "https://")):
            raise ToolError(
                ToolErrorType.UNSUPPORTED,
                f"仅支持 http/https 链接，收到: {url[:80]}",
            )

        last_exc: Optional[ToolError] = None
        for attempt in range(self.max_retries + 1):
            try:
                status, html = _http_request(url, timeout=self.timeout)
                text = self._extract_text(html)
                text = self._clean(text)
                if len(text) < 50:
                    return (
                        f"页面 {url} 已打开，但未抽取到有效正文（可能是登录页、"
                        "纯脚本页或被反爬拦截）。请尝试其他搜索结果。"
                    )
                if len(text) > self.max_chars:
                    text = text[: self.max_chars] + "\n...[网页正文过长，已截断]"
                return f"网页 {url} 的正文：\n\n{text}"
            except ToolError as exc:
                last_exc = exc
                if not exc.retryable:
                    raise
                if attempt < self.max_retries:
                    time.sleep(min(self.backoff_base ** attempt, 8.0))
                continue
        if last_exc is not None:
            raise last_exc
        raise ToolError(ToolErrorType.UNKNOWN, "打开网页重试耗尽")

    # ------------------------------------------------------------------
    @staticmethod
    def _extract_text(html: str) -> str:
        # 1) trafilatura 质量最好
        try:
            import trafilatura  # type: ignore

            text = trafilatura.extract(html, include_comments=False, include_tables=True)
            if text:
                return text
        except Exception:
            pass
        # 2) BeautifulSoup
        try:
            from bs4 import BeautifulSoup  # type: ignore

            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()
            return soup.get_text(separator="\n")
        except Exception:
            pass
        # 3) 标准库兜底
        return OpenTool._strip_html(html)

    @staticmethod
    def _strip_html(html: str) -> str:
        html = re.sub(r"(?is)<(script|style|nav|footer|header)[^>]*>.*?</\1>", " ", html)
        html = re.sub(r"(?s)<br\s*/?>", "\n", html)
        html = re.sub(r"(?s)</p>", "\n", html)
        text = re.sub(r"(?s)<[^>]+>", " ", html)
        return text

    @staticmethod
    def _clean(text: str) -> str:
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        # HTML 实体（常见的几个）
        for entity, ch in (("&nbsp;", " "), ("&amp;", "&"), ("&quot;", '"'),
                           ("&#39;", "'"), ("&lt;", "<"), ("&gt;", ">")):
            text = text.replace(entity, ch)
        return text.strip()


__all__ = ["OpenTool"]
