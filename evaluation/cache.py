# =============================================================================
# evaluation/cache.py — 评测缓存（JSON 落盘）
# =============================================================================
# 目的：
#   1. LLM 报告生成是评测中唯一昂贵步骤，缓存后支持断点续跑与重跑；
#   2. Claim 抽取 / Judge 结果缓存，避免重复计算。
#
# 用法：
#   store = CacheStore(cache_dir, enabled=True)
#   store.get("report_M001_2025-01")   # -> dict | None
#   store.set("report_M001_2025-01", report)
# =============================================================================

import hashlib
import json
import os
import threading
from typing import Any, Optional


class CacheStore:
    """基于 JSON 文件的缓存。key 自动做路径安全化。"""

    def __init__(self, cache_dir: str, enabled: bool = True):
        self.cache_dir = cache_dir
        self.enabled = enabled
        self._lock = threading.Lock()
        if enabled:
            os.makedirs(cache_dir, exist_ok=True)

    def _path(self, key: str) -> str:
        safe = hashlib.md5(key.encode("utf-8")).hexdigest()[:16]
        # 保留可读前缀便于人工排查
        prefix = "".join(c if c.isalnum() else "_" for c in key)[:60] or "item"
        return os.path.join(self.cache_dir, f"{prefix}_{safe}.json")

    def get(self, key: str) -> Optional[Any]:
        if not self.enabled:
            return None
        path = self._path(key)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def set(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        path = self._path(key)
        with self._lock:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(value, f, ensure_ascii=False)
            except (TypeError, OSError):
                pass

    def clear(self) -> int:
        """清空缓存，返回删除文件数。"""
        if not os.path.isdir(self.cache_dir):
            return 0
        n = 0
        for name in os.listdir(self.cache_dir):
            if name.endswith(".json"):
                try:
                    os.remove(os.path.join(self.cache_dir, name))
                    n += 1
                except OSError:
                    pass
        return n
