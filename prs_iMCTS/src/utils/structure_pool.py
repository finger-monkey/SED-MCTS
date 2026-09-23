from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional
import hashlib
import re


@dataclass
class PoolItem:
    expr_str: str
    reward: float
    path: List[str]
    metadata: Optional[Dict[str, Any]] = None


class DiversityPool:


    def __init__(self) -> None:
        self._items: Dict[str, PoolItem] = {}

    @staticmethod
    def canonicalize_structure(expr_str: str) -> str:
        s = expr_str
        s = re.sub(r"R\[\d+\]", "R", s)
        s = re.sub(r"C\[\d+\]", "C", s)
        s = re.sub(r"(?<![A-Za-z_])[-+]?\d+(\.\d+)?([eE][-+]?\d+)?", "K", s)
        s = re.sub(r"\s+", "", s)
        return s

    def structure_hash(self, expr_str: str) -> str:
        canonical = self.canonicalize_structure(expr_str)
        return hashlib.md5(canonical.encode("utf-8")).hexdigest()

    def upsert(
        self,
        expr_str: str,
        reward: float,
        path: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        key = self.structure_hash(expr_str)
        old = self._items.get(key)
        if old is None or float(reward) > old.reward:
            self._items[key] = PoolItem(
                expr_str=expr_str,
                reward=float(reward),
                path=list(path),
                metadata=metadata,
            )
            return True
        return False

    def __len__(self) -> int:
        return len(self._items)

    def iter_representatives(self) -> Iterable[PoolItem]:
        return self._items.values()

    def topk(self, k: int) -> List[PoolItem]:
        k = max(0, int(k))
        ranked = sorted(self._items.values(), key=lambda x: x.reward, reverse=True)
        return ranked[:k]
