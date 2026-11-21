"""用真实 Ollama embedding 验证长期记忆的召回效果（不调 LLM，几秒钟）。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.memory.long_memory import get_long_memory  # noqa: E402

FACTS = [
    {"kind": "岗位", "text": "岗位：市场岗"},
    {"kind": "入职时间", "text": "入职时间：2023年3月"},
    {"kind": "常驻城市", "text": "常驻城市：长沙"},
    {"kind": "部门", "text": "部门：商务部"},
]

QUESTIONS = [
    "我之前说过什么岗位？",
    "我什么时候入职的？",
    "我的住宿标准该按哪一档？",
    "我在哪个城市？",
    "食堂几点开饭？",          # 应该召不回任何记忆
]

mem = get_long_memory()
for row in mem._rows():
    try:
        mem.delete(row["id"])
    except Exception:  # noqa: BLE001
        pass

uid, sid = "default", "sess-livecheck"
added = mem.remember(uid, sid, FACTS)
print(f"写入 {added} 条，库里共 {mem.count(uid)} 条\n")

for q in QUESTIONS:
    hits = mem.recall(uid, q)
    print(f"问：{q}")
    if not hits:
        print("  → 无召回")
    for h in hits:
        print(f"  → {h['text']}  cos={h['cos']}  {h['created_at']}")
    print()

# 看每个问题的最高分，便于判断阈值是否合适
print("=== 原始 top1 分数（不过滤）===")
for q in QUESTIONS:
    vec = mem._query_embed(q)
    res = mem.store.query([vec], n_results=2)
    dists = (res.get("distances") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    if dists:
        print(f"{q} → cos={round(1 - float(dists[0]), 3)} / {metas[0].get('kind')}")

# 清理测试数据
for row in mem._rows():
    try:
        mem.delete(row["id"])
    except Exception:  # noqa: BLE001
        pass
print("\n测试数据已清理，剩余", mem.count(), "条")
