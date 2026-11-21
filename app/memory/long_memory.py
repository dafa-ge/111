"""长期向量记忆：把用户稳定事实写成向量，提问时按语义召回。

要点：
1. 与制度库**物理隔离**（SimpleStore 用 data/memory_vectors，Chroma 用独立集合 user_memory），
   避免"用户是市场岗"被当成制度条款召回。
2. 幂等：事实文本 sha256 取前 16 位做 id，重复说不堆新条目。
3. 同 kind 只留最新：换岗后旧岗位会被删掉，不会互相打架。
4. 30 天过期：检索时按 created_at 过滤，另有 purge_expired 手动清理。
"""
from __future__ import annotations

import hashlib
import threading
from datetime import datetime, timedelta

from ..rag import ollama_client
from ..rag.config import (
    MEMORY_COLLECTION, MEMORY_INJECT_ALL_LIMIT, MEMORY_MAX_CHARS, MEMORY_MIN_COS,
    MEMORY_TOP_K, MEMORY_TTL_DAYS, MEMORY_VECTOR_DIR,
)
from ..rag.vector_store import ChromaStore, SimpleStore, decide_backend


def _fingerprint(user_id: str, text: str) -> str:
    return hashlib.sha256(f"{user_id}|{text}".encode("utf-8")).hexdigest()[:16]


def _cutoff(days: int) -> str:
    return (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S")


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


class LongMemory:
    def __init__(
        self,
        store=None,
        embed=None,
        query_embed=None,
        ttl_days: int = MEMORY_TTL_DAYS,
        top_k: int = MEMORY_TOP_K,
        min_cos: float = MEMORY_MIN_COS,
        max_chars: int = MEMORY_MAX_CHARS,
        inject_all_limit: int = MEMORY_INJECT_ALL_LIMIT,
    ):
        self._store = store
        self._embed = embed or ollama_client.embed_texts
        self._query_embed = query_embed or ollama_client.embed_query
        self.ttl_days = ttl_days
        self.top_k = top_k
        self.min_cos = min_cos
        self.max_chars = max_chars
        self.inject_all_limit = inject_all_limit
        self._lock = threading.Lock()

    @property
    def store(self):
        if self._store is None:
            backend, _ = decide_backend()
            if backend == "chroma":
                self._store = ChromaStore(collection=MEMORY_COLLECTION)
            else:
                self._store = SimpleStore(MEMORY_VECTOR_DIR)
        return self._store

    # ---------------------------------------------------------------- 读原始行
    def _rows(self, user_id: str | None = None) -> list[dict]:
        try:
            data = self.store.get(limit=100000)
        except TypeError:
            data = self.store.get()
        ids = data.get("ids") or []
        docs = data.get("documents") or []
        metas = data.get("metadatas") or []
        rows = []
        for i, mid in enumerate(ids):
            meta = metas[i] or {}
            if user_id and meta.get("user_id") != user_id:
                continue
            rows.append({"id": mid, "text": docs[i], "meta": meta})
        return rows

    # ---------------------------------------------------------------- 写入
    def remember(self, user_id: str, session_id: str, facts: list[dict]) -> int:
        if not facts:
            return 0
        with self._lock:
            texts = [f["text"][: self.max_chars] for f in facts]
            ids = [_fingerprint(user_id, t) for t in texts]
            kinds = {f["kind"] for f in facts}

            # 同一 kind 的旧条目删掉（换岗、改部门这类），只保留最新
            for row in self._rows(user_id):
                if row["meta"].get("kind") in kinds and row["id"] not in ids:
                    try:
                        self.store.delete(where={"mem_id": row["id"]})
                    except Exception:  # noqa: BLE001
                        pass

            embeddings = self._embed(texts)
            stamp = _now()
            metadatas = [
                {
                    "mem_id": ids[i],
                    "user_id": user_id,
                    "session_id": session_id,
                    "kind": facts[i]["kind"],
                    "text": texts[i],
                    "created_at": stamp,
                    "source": "memory",
                }
                for i in range(len(texts))
            ]
            self.store.upsert(ids, texts, embeddings, metadatas)
        return len(texts)

    # ---------------------------------------------------------------- 召回
    def recall(self, user_id: str, question: str, k: int | None = None,
               inject_all_limit: int | None = None) -> list[dict]:
        """召回记忆。

        策略：库里有效记忆少时**全量注入**（只按相关度排序），多了才按阈值检索。
        原因：bge-m3 对短事实句的余弦区分度低，实测"我的住宿标准该按哪一档"
        对"岗位：市场岗"只有 0.43，纯阈值检索会漏掉最该用的那条。
        """
        k = k or self.top_k
        limit = inject_all_limit or self.inject_all_limit
        if not (question or "").strip():
            return []

        vec = self._query_embed(question)
        n = max(k * 4, limit, 8)
        res = self.store.query([vec], n_results=n)
        ids = (res.get("ids") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        cos_by_id = {mid: 1.0 - float(d) for mid, d in zip(ids, dists)}

        cutoff = _cutoff(self.ttl_days)
        live = [r for r in self._rows(user_id) if (r["meta"].get("created_at") or "") >= cutoff]

        if live and len(live) <= limit:
            # 全量注入：不设相关度门槛，只做同 kind 去重和排序
            items = [
                {
                    "id": r["id"],
                    "text": r["text"],
                    "kind": r["meta"].get("kind"),
                    "cos": round(cos_by_id.get(r["id"], 0.0), 3),
                    "created_at": r["meta"].get("created_at"),
                    "mode": "all",
                }
                for r in live
            ]
            latest: dict[str, dict] = {}
            for it in items:
                cur = latest.get(it["kind"])
                if cur is None or (it["created_at"] or "") > (cur["created_at"] or ""):
                    latest[it["kind"]] = it
            final = sorted(latest.values(), key=lambda x: x["cos"], reverse=True)[:limit]
            for p in final:
                p["text"] = p["text"][: self.max_chars]
            return final

        # 记忆较多：按阈值检索
        ids_q = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists_q = (res.get("distances") or [[]])[0]
        picked: list[dict] = []
        for i, mid in enumerate(ids_q):
            meta = metas[i] or {}
            cos = 1.0 - float(dists_q[i])
            if meta.get("user_id") != user_id:
                continue
            if cos < self.min_cos:
                continue
            if (meta.get("created_at") or "") < cutoff:
                continue
            picked.append({
                "id": mid,
                "text": docs[i],
                "kind": meta.get("kind"),
                "cos": round(cos, 3),
                "created_at": meta.get("created_at"),
                "mode": "retrieval",
            })

        latest = {}
        for p in picked:
            cur = latest.get(p["kind"])
            if cur is None or (p["created_at"] or "") > (cur["created_at"] or ""):
                latest[p["kind"]] = p

        final = sorted(latest.values(), key=lambda x: x["cos"], reverse=True)[:k]
        for p in final:
            p["text"] = p["text"][: self.max_chars]
        return final

    # ---------------------------------------------------------------- 管理
    def list_all(self, user_id: str) -> list[dict]:
        rows = [r for r in self._rows(user_id) if r["meta"].get("source") == "memory"]
        rows.sort(key=lambda r: r["meta"].get("created_at") or "", reverse=True)
        return [
            {
                "id": r["id"],
                "text": r["text"],
                "kind": r["meta"].get("kind"),
                "created_at": r["meta"].get("created_at"),
                "session_id": r["meta"].get("session_id"),
            }
            for r in rows
        ]

    def delete(self, mem_id: str) -> bool:
        try:
            self.store.delete(where={"mem_id": mem_id})
            return True
        except Exception:  # noqa: BLE001
            return False

    def purge_expired(self, days: int | None = None, user_id: str | None = None) -> int:
        """清理过期记忆。user_id=None 清全库（管理员），指定则只清该用户。"""
        cutoff = _cutoff(days or self.ttl_days)
        removed = 0
        for row in self._rows(user_id):
            if (row["meta"].get("created_at") or "") < cutoff:
                try:
                    self.store.delete(where={"mem_id": row["id"]})
                    removed += 1
                except Exception:  # noqa: BLE001
                    continue
        return removed

    def count(self, user_id: str | None = None) -> int:
        return len(self._rows(user_id))


_long: LongMemory | None = None


def get_long_memory() -> LongMemory:
    global _long
    if _long is None:
        _long = LongMemory()
    return _long
