"""混合检索器：BM25 关键词召回 + 向量语义召回，RRF 融合，邻接扩展，低置信兜底。"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass

import jieba
import numpy as np
from rank_bm25 import BM25Okapi

from .config import (
    MIN_BM25, MIN_COS_SIM, NEIGHBOR_BONUS, NEIGHBOR_ENABLED,
    RRF_K, STRONG_COS_SIM, TOP_K_BM25, TOP_K_VEC, TOP_N_FINAL,
)
from .ollama_client import embed_query

jieba.setLogLevel(60)

# 只过滤虚词与纯连接词；业务词（岗位、省内、住宿、报销、标准）一律保留
STOPWORDS = set(
    """的 了 是 在 和 与 或 及 对 为 上 下 中 有 无 不 也 都 就 而 但 并 被 把 从 到 由 于
    这 那 其 该 本 此 个 些 等 以 将 已 可 能 应 须 若 如 则 按照 根据 对于 关于 通过
    进行 一个 以及 及其 其中 相关 有关 上述 如下 以下 什么 多少 怎么 如何 是否 吗 呢 啊
    公司 员工 可以 需要 我们 你们 他们 它""".split()
)


def normalize(text: str) -> str:
    """全角转半角、统一大小写。入库与查询两侧都走这里，保证"３００"能搜到"300"。"""
    return unicodedata.normalize("NFKC", text).lower()


def tokenize(text: str) -> list[str]:
    result: list[str] = []
    for tok in jieba.lcut(normalize(text), cut_all=False):
        tok = tok.strip()
        if not tok or tok in STOPWORDS:
            continue
        if len(tok) == 1 and not tok.isdigit():
            continue
        result.append(tok)
    return result


@dataclass
class Hit:
    chunk_id: str
    text: str
    source: str
    page: int
    section: str = ""
    cos_sim: float = 0.0
    bm25: float = 0.0
    rrf: float = 0.0
    neighbor: bool = False

    def head(self, width: int = 60) -> str:
        one = self.text.replace("\n", " ")
        return one[:width] + ("…" if len(one) > width else "")


class HybridRetriever:
    def __init__(self, collection, chunks: list[dict]):
        self.col = collection
        self.chunks = chunks
        self.by_id = {c["chunk_id"]: c for c in chunks}
        self.idx_map = {
            (c["source"], int(c["chunk_index"])): c["chunk_id"] for c in chunks
        }
        self._tokens = [(tokenize(c["text"]) or ["空"]) for c in chunks]
        self.bm25 = BM25Okapi(self._tokens, k1=1.5, b=0.75) if self._tokens else None

    # ---- 两路召回 ----
    def _vector_hits(self, query: str, k: int) -> list[tuple[str, float]]:
        if not self.chunks:
            return []
        qv = embed_query(query)
        n = max(1, min(k, len(self.chunks)))
        res = self.col.query(
            query_embeddings=[qv], n_results=n, include=["distances", "metadatas"]
        )
        hits: list[tuple[str, float]] = []
        for cid, dist in zip(res["ids"][0], res["distances"][0]):
            hits.append((cid, max(0.0, 1.0 - float(dist))))
        return hits

    def _bm25_hits(self, query: str, k: int) -> list[tuple[str, float]]:
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(scores)[::-1][:k]
        return [
            (self.chunks[i]["chunk_id"], float(scores[i]))
            for i in order
            if scores[i] > 0
        ]

    # ---- 融合 ----
    def search(self, query: str, top_n: int = TOP_N_FINAL) -> list[Hit]:
        vec = self._vector_hits(query, TOP_K_VEC)
        kw = self._bm25_hits(query, TOP_K_BM25)

        rrf: dict[str, float] = {}
        for rank, (cid, _) in enumerate(vec):
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, (cid, _) in enumerate(kw):
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)

        cos = dict(vec)
        bm = dict(kw)

        hits: list[Hit] = []
        for cid, score in sorted(rrf.items(), key=lambda x: -x[1])[:top_n]:
            c = self.by_id.get(cid)
            if not c:
                continue
            hits.append(
                Hit(
                    chunk_id=cid,
                    text=c["text"],
                    source=c["source"],
                    page=int(c["page"]),
                    section=c.get("section", ""),
                    cos_sim=cos.get(cid, 0.0),
                    bm25=bm.get(cid, 0.0),
                    rrf=score,
                )
            )
        if NEIGHBOR_ENABLED:
            hits = self._expand(hits)
        return hits

    def _expand(self, hits: list[Hit]) -> list[Hit]:
        """邻接扩展：把同文件前后各 1 块带进上下文，解决"标准"与"例外"分块的问题。"""
        existing = {h.chunk_id for h in hits}
        extra: list[Hit] = []
        for h in hits:
            if h.neighbor:
                continue
            c = self.by_id.get(h.chunk_id)
            if not c:
                continue
            base = int(c["chunk_index"])
            for delta in (-1, 1):
                nid = self.idx_map.get((c["source"], base + delta))
                if not nid or nid in existing:
                    continue
                nc = self.by_id[nid]
                extra.append(
                    Hit(
                        chunk_id=nid,
                        text=nc["text"],
                        source=nc["source"],
                        page=int(nc["page"]),
                        section=nc.get("section", ""),
                        cos_sim=h.cos_sim * NEIGHBOR_BONUS,
                        bm25=h.bm25 * NEIGHBOR_BONUS,
                        rrf=h.rrf * NEIGHBOR_BONUS,
                        neighbor=True,
                    )
                )
                existing.add(nid)
        return hits + extra


def is_confident(hits: list[Hit]) -> bool:
    """双条件兜底：强语义直接算相关；中等语义必须**同一个片段**同时有 BM25 关键词支撑。

    小库场景下 bge-m3 对任何中文问题都会给出 0.4~0.5 的相似度，
    单靠绝对阈值必然放过无关问题，所以必须叠加关键词条件。

    注意：余弦和 BM25 必须取自同一个片段。早期版本分别取 max，
    结果"食堂几点开饭"拿考勤办法的 0.506 余弦去配雨量计说明书的 4.33 BM25，
    两个不相干的片段拼成"可信"，导致该拒答的没拒答。
    """
    for h in hits:
        if h.cos_sim >= STRONG_COS_SIM:
            return True
        if h.cos_sim >= MIN_COS_SIM and h.bm25 >= MIN_BM25:
            return True
    return False
