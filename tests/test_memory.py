"""记忆模块单测：不依赖 Ollama，用假 store / 假 embedding 跑。

运行：python tests/test_memory.py
"""
from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.memory.extractor import extract, should_llm_fallback  # noqa: E402
from app.memory.long_memory import LongMemory  # noqa: E402
from app.memory.short_memory import ShortMemory  # noqa: E402


class FakeStore:
    """内存版向量库，接口对齐 SimpleStore。"""

    def __init__(self):
        self.ids: list[str] = []
        self.documents: list[str] = []
        self.metadatas: list[dict] = []
        self.vectors: list[list[float]] = []

    def upsert(self, ids, documents, embeddings, metadatas):
        for i, mid in enumerate(ids):
            if mid in self.ids:
                idx = self.ids.index(mid)
                self.documents[idx] = documents[i]
                self.metadatas[idx] = metadatas[i]
                self.vectors[idx] = list(embeddings[i])
            else:
                self.ids.append(mid)
                self.documents.append(documents[i])
                self.metadatas.append(metadatas[i])
                self.vectors.append(list(embeddings[i]))

    def delete(self, where: dict):
        keep = [i for i, m in enumerate(self.metadatas)
                if not all(m.get(k) == v for k, v in where.items())]
        self.ids = [self.ids[i] for i in keep]
        self.documents = [self.documents[i] for i in keep]
        self.metadatas = [self.metadatas[i] for i in keep]
        self.vectors = [self.vectors[i] for i in keep]

    def count(self):
        return len(self.ids)

    def get(self, limit=100000, include=None):
        return {"ids": self.ids, "documents": self.documents,
                "metadatas": self.metadatas}

    def query(self, query_embeddings, n_results=3, include=None):
        q = list(query_embeddings[0])
        scored = []
        for i, v in enumerate(self.vectors):
            sim = sum(a * b for a, b in zip(q, v))
            scored.append((i, 1.0 - sim))
        scored.sort(key=lambda x: x[1])
        chosen = scored[:n_results]
        return {
            "ids": [[self.ids[i] for i, _ in chosen]],
            "documents": [[self.documents[i] for i, _ in chosen]],
            "metadatas": [[self.metadatas[i] for i, _ in chosen]],
            "distances": [[d for _, d in chosen]],
        }


def fake_embed(texts):
    """把文本映射成稳定 4 维向量：含关键词就偏向对应维度。"""
    out = []
    for t in texts:
        v = [0.0, 0.0, 0.0, 0.0]
        if "岗位" in t or "市场" in t or "技术" in t:
            v[0] = 1.0
        if "入职" in t:
            v[1] = 1.0
        if "城市" in t:
            v[2] = 1.0
        if "称呼" in t:
            v[3] = 1.0
        if not any(v):
            v[0] = 0.5
        out.append(v)
    return out


def fake_query(text):
    return fake_embed([text])[0]


class TestExtractor(unittest.TestCase):
    def test_role(self):
        facts = extract("我是市场岗，负责华东区域的商务")
        self.assertTrue(any(f["kind"] == "岗位" and "市场" in f["text"] for f in facts))

    def test_hire_date(self):
        facts = extract("我2023年3月入职的")
        hit = [f for f in facts if f["kind"] == "入职时间"]
        self.assertTrue(hit)
        self.assertIn("2023", hit[0]["text"])

    def test_city(self):
        facts = extract("我常驻长沙")
        self.assertTrue(any(f["kind"] == "常驻城市" for f in facts))

    def test_no_fact(self):
        self.assertEqual(extract("省内出差住宿标准是多少？"), [])

    def test_literal_guard(self):
        """模型编出来的值（原话里没有）必须被丢掉。"""
        self.assertEqual(extract("我是火星岗"), [])
        self.assertFalse(should_llm_fallback("省内出差住宿标准是多少？"))


class TestShortMemory(unittest.TestCase):
    def test_window(self):
        mem = ShortMemory(max_messages=4, persist_dir=None)
        for i in range(6):
            mem.append("sess-abc123", "user", f"q{i}")
        self.assertEqual(len(mem.history("sess-abc123")), 4)
        self.assertEqual(mem.history("sess-abc123")[-1]["content"], "q5")

    def test_bad_id(self):
        mem = ShortMemory(persist_dir=None)
        with self.assertRaises(ValueError):
            mem.append("../evil", "user", "x")

    def test_clear(self):
        mem = ShortMemory(persist_dir=None)
        mem.append("sess-abc123", "user", "q")
        mem.clear("sess-abc123")
        self.assertEqual(mem.history("sess-abc123"), [])


class TestLongMemory(unittest.TestCase):
    def setUp(self):
        self.store = FakeStore()
        self.mem = LongMemory(store=self.store, embed=fake_embed,
                              query_embed=fake_query, ttl_days=30)

    def test_idempotent(self):
        self.mem.remember("default", "sess-abc123", [{"kind": "岗位", "text": "岗位：市场岗"}])
        self.mem.remember("default", "sess-abc123", [{"kind": "岗位", "text": "岗位：市场岗"}])
        self.assertEqual(self.mem.count("default"), 1)

    def test_same_kind_replaced(self):
        self.mem.remember("default", "sess-abc123", [{"kind": "岗位", "text": "岗位：市场岗"}])
        self.mem.remember("default", "sess-abc123", [{"kind": "岗位", "text": "岗位：技术岗"}])
        self.assertEqual(self.mem.count("default"), 1)
        self.assertIn("技术", self.mem.list_all("default")[0]["text"])

    def test_recall(self):
        self.mem.remember("default", "sess-abc123", [{"kind": "岗位", "text": "岗位：市场岗"}])
        hits = self.mem.recall("default", "岗位相关的问题")
        self.assertTrue(hits)
        self.assertIn("市场岗", hits[0]["text"])

    def test_user_isolation(self):
        self.mem.remember("alice", "sess-abc123", [{"kind": "岗位", "text": "岗位：市场岗"}])
        self.assertEqual(self.mem.recall("bob", "岗位"), [])
        self.assertEqual(self.mem.count("bob"), 0)

    def test_ttl_filter(self):
        self.mem.remember("default", "sess-abc123", [{"kind": "岗位", "text": "岗位：市场岗"}])
        old = (datetime.now() - timedelta(days=45)).strftime("%Y-%m-%dT%H:%M:%S")
        self.store.metadatas[0]["created_at"] = old
        self.assertEqual(self.mem.recall("default", "岗位"), [])
        self.assertEqual(self.mem.purge_expired(), 1)
        self.assertEqual(self.mem.count(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
