"""环境自检：Ollama 可用性、模型是否就位、后端选择、库状态一致性。不会触发 Chroma 写入。"""
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.rag import vector_store  # noqa: E402
from app.rag.config import (  # noqa: E402
    CHUNKS_JSON, EMBED_MODEL, LLM_MODEL, OLLAMA_BASE,
)


def main():
    print("== Ollama ==")
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE}/api/tags", timeout=8) as r:
            models = [m["name"] for m in json.loads(r.read().decode())["models"]]
        print("  可用：", models)
        for need in (EMBED_MODEL, LLM_MODEL):
            base = need.split(":")[0]
            ok = any(m.split(":")[0] == base for m in models)
            print(f"  {need}: {'OK' if ok else '缺失，需 ollama pull ' + need}")
    except Exception as e:  # noqa: BLE001
        print("  不可用：", e)

    print("\n== 向量后端 ==")
    backend, reason = vector_store.decide_backend()
    print(f"  当前：{backend}")
    print(f"  原因：{reason}")

    print("\n== 库状态 ==")
    chunks = 0
    if Path(CHUNKS_JSON).exists():
        chunks = len(json.loads(Path(CHUNKS_JSON).read_text(encoding="utf-8")))
    store = vector_store.get_store()
    n = store.count()
    print(f"  chunks.json：{chunks}")
    print(f"  向量库：{n}")
    print(f"  一致性：{'OK' if chunks == n else '不一致，需要 build --rebuild'}")


if __name__ == "__main__":
    main()
