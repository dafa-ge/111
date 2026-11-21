"""向量库封装：统一接口，双后端。

- ChromaStore：chromadb 持久化（默认首选，data/chroma_db）
- SimpleStore：内置 numpy 余弦检索（data/vector_data），当 Chroma 在本机不可用时兜底

接口保持与 chromadb Collection 一致：upsert / delete / count / query，
所以上层（ingest、retriever）不需要关心用的是哪个后端。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import (
    BACKEND_DECISION, CHROMA_DIR, COLLECTION_NAME, VECTOR_BACKEND, VECTOR_DATA_DIR,
)
from .splitter import Chunk


class SimpleStore:
    """纯 numpy 的余弦检索，持久化两个文件，适合万条以内的规模。"""

    def __init__(self, data_dir: Path | str = VECTOR_DATA_DIR):
        self.dir = Path(data_dir)
        self.ids: list[str] = []
        self.documents: list[str] = []
        self.metadatas: list[dict] = []
        self.embeddings: np.ndarray = np.zeros((0, 1024), dtype=np.float32)
        self._load()

    # ---- 持久化 ----
    def _load(self) -> None:
        meta_path = self.dir / "meta.json"
        emb_path = self.dir / "embeddings.npy"
        if meta_path.exists() and emb_path.exists():
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            self.ids = data.get("ids", [])
            self.documents = data.get("documents", [])
            self.metadatas = data.get("metadatas", [])
            self.embeddings = np.load(emb_path)

    def _save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        if self.embeddings.size:
            np.save(self.dir / "embeddings.npy", self.embeddings)
        else:
            path = self.dir / "embeddings.npy"
            if path.exists():
                path.unlink()
        meta = {
            "ids": self.ids,
            "documents": self.documents,
            "metadatas": self.metadatas,
            "dim": int(self.embeddings.shape[1]) if self.embeddings.size else 0,
        }
        tmp = self.dir / "meta.json.tmp"
        tmp.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.dir / "meta.json")

    def reset(self) -> None:
        self.ids, self.documents, self.metadatas = [], [], []
        self.embeddings = np.zeros((0, 0), dtype=np.float32)
        self._save()

    # ---- 写入 ----
    def upsert(self, ids, documents, embeddings, metadatas) -> None:
        new_vecs = np.asarray(embeddings, dtype=np.float32)
        new_vecs = self._normalize(new_vecs)
        if self.embeddings.size == 0:
            self.embeddings = new_vecs
            self.ids = list(ids)
            self.documents = list(documents)
            self.metadatas = list(metadatas)
            self._save()
            return

        index = {cid: i for i, cid in enumerate(self.ids)}
        vecs = list(self.embeddings)
        for j, cid in enumerate(ids):
            if cid in index:
                i = index[cid]
                vecs[i] = new_vecs[j]
                self.documents[i] = documents[j]
                self.metadatas[i] = metadatas[j]
            else:
                index[cid] = len(self.ids)
                self.ids.append(cid)
                self.documents.append(documents[j])
                self.metadatas.append(metadatas[j])
                vecs.append(new_vecs[j])
        self.embeddings = np.vstack(vecs)
        self._save()

    @staticmethod
    def _normalize(m: np.ndarray) -> np.ndarray:
        if m.size == 0:
            return m
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return m / norms

    def delete(self, where: dict) -> None:
        keep = [
            i
            for i, meta in enumerate(self.metadatas)
            if not all(meta.get(k) == v for k, v in where.items())
        ]
        if len(keep) == len(self.ids):
            return
        self.ids = [self.ids[i] for i in keep]
        self.documents = [self.documents[i] for i in keep]
        self.metadatas = [self.metadatas[i] for i in keep]
        self.embeddings = self.embeddings[keep] if self.embeddings.size else self.embeddings
        self._save()

    def count(self) -> int:
        return len(self.ids)

    # ---- 查询 ----
    def query(self, query_embeddings, n_results: int, include=None) -> dict:
        if not self.ids:
            return {"ids": [[]], "distances": [[]], "documents": [[]], "metadatas": [[]]}
        q = self._normalize(np.asarray(query_embeddings, dtype=np.float32))[0]
        sims = self.embeddings @ q
        order = np.argsort(sims)[::-1][: max(1, min(n_results, len(self.ids)))]
        return {
            "ids": [[self.ids[i] for i in order]],
            "distances": [[float(1.0 - sims[i]) for i in order]],   # 与 Chroma cosine 距离同义
            "documents": [[self.documents[i] for i in order]],
            "metadatas": [[self.metadatas[i] for i in order]],
        }

    def get(self, limit: int = 3, include=None) -> dict:
        return {
            "ids": self.ids[:limit],
            "documents": self.documents[:limit],
            "metadatas": self.metadatas[:limit],
        }


class ChromaStore:
    """chromadb 封装，接口与 SimpleStore 一致。"""

    def __init__(self, recreate: bool = False, collection: str = COLLECTION_NAME):
        import chromadb

        self.collection = collection
        Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        if recreate:
            try:
                self.client.delete_collection(collection)
            except Exception:  # noqa: BLE001
                pass
        try:
            self.col = self.client.get_or_create_collection(
                name=collection, metadata={"hnsw:space": "cosine"}
            )
        except Exception:  # noqa: BLE001
            self.col = self.client.get_or_create_collection(name=collection)

    def upsert(self, ids, documents, embeddings, metadatas) -> None:
        self.col.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

    def delete(self, where: dict) -> None:
        self.col.delete(where=where)

    def count(self) -> int:
        return self.col.count()

    def get(self, limit: int = 3, include=None) -> dict:
        return self.col.get(limit=limit, include=include or ["documents", "metadatas"])

    def query(self, query_embeddings, n_results: int, include=None) -> dict:
        return self.col.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            include=include or ["documents", "metadatas", "distances"],
        )


def decide_backend() -> tuple[str, str]:
    """返回 (后端名, 原因)。decision 文件由 build 首次生成；没有则默认 simple。"""
    if VECTOR_BACKEND in {"chroma", "simple"}:
        return VECTOR_BACKEND, "由配置或环境变量指定"
    if BACKEND_DECISION.exists():
        try:
            data = json.loads(BACKEND_DECISION.read_text(encoding="utf-8"))
            return data.get("backend", "simple"), data.get("reason", "")
        except Exception:  # noqa: BLE001
            pass
    return "simple", "未做过后端探测，默认使用内置向量后端"


def record_backend(backend: str, reason: str) -> None:
    Path(BACKEND_DECISION).parent.mkdir(parents=True, exist_ok=True)
    Path(BACKEND_DECISION).write_text(
        json.dumps({"backend": backend, "reason": reason}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def get_store(recreate: bool = False):
    backend, _ = decide_backend()
    if backend == "chroma":
        return ChromaStore(recreate=recreate)
    store = SimpleStore()
    if recreate:
        store.reset()
    return store


def backend_name() -> str:
    return decide_backend()[0]


# ---- 兼容旧调用（ingest 用）----
def upsert_chunks(store, chunks: list[Chunk], embeddings: list[list[float]], batch_size: int = 32) -> int:
    total = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        store.upsert(
            ids=[c.chunk_id for c in batch],
            documents=[c.text for c in batch],
            embeddings=embeddings[i : i + batch_size],
            metadatas=[
                {
                    "source": c.source,
                    "file_type": c.file_type,
                    "page": int(c.page),
                    "chunk_index": int(c.chunk_index),
                    "section": c.section,
                }
                for c in batch
            ],
        )
        total += len(batch)
    return total


def delete_source(store, source: str) -> None:
    try:
        store.delete(where={"source": source})
    except Exception:  # noqa: BLE001
        pass


def count(store) -> int:
    try:
        return store.count()
    except Exception:  # noqa: BLE001
        return 0
