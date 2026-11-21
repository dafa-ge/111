"""服务装配：从磁盘读取 chunks.json 与 Chroma，构造检索器。"""
from __future__ import annotations

import json
from pathlib import Path

from . import vector_store
from .config import CHUNKS_JSON
from .retriever import HybridRetriever


def load_chunks(path: Path = CHUNKS_JSON) -> list[dict]:
    if not Path(path).exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_retriever() -> HybridRetriever:
    chunks = load_chunks()
    if not chunks:
        raise RuntimeError("chunks.json 为空，请先运行：python -m app.rag.build")
    col = vector_store.get_store()
    n = vector_store.count(col)
    if n != len(chunks):
        raise RuntimeError(
            f"向量库有 {n} 条，chunks.json 有 {len(chunks)} 条，两边不同步。"
            "请重跑 build（或加 --rebuild）。"
        )
    return HybridRetriever(col, chunks)
