"""入库编排：文件指纹幂等 + 稳定 ID + 增量更新 + 一致性校验。"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

from . import loader, ollama_client, splitter, vector_store
from .config import CHUNKS_JSON, DOCS_DIR, MANIFEST_JSON


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path, default):
    if not Path(path).exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_manifest() -> dict:
    return load_json(MANIFEST_JSON, {})


def load_chunks() -> list[dict]:
    return load_json(CHUNKS_JSON, [])


def ingest(docs_dir: Path | str | None = None, rebuild: bool = False) -> dict:
    docs_dir = Path(docs_dir or DOCS_DIR)
    stats: dict = {
        "docs_dir": str(docs_dir),
        "rebuild": rebuild,
        "new": [],
        "updated": [],
        "skipped": [],
        "removed": [],
        "empty_files": [],
        "failed": [],
        "errors": [],
        "chunks_added": 0,
    }

    prev_manifest: dict = {} if rebuild else load_manifest()
    prev_chunks: list[dict] = [] if rebuild else load_chunks()

    client = None
    col = vector_store.get_store(recreate=rebuild)

    load_report = loader.load_directory(docs_dir)
    stats["empty_files"] = load_report.empty_files
    stats["failed"] = load_report.failed
    stats["ignored"] = load_report.ignored

    docs = {d.source: d for d in load_report.docs}
    hashes = {}
    for src, doc in docs.items():
        try:
            hashes[src] = file_sha256(Path(doc.path))
        except Exception as e:  # noqa: BLE001
            stats["errors"].append(f"{src}：读取失败 {e}")
            docs.pop(src)

    kept_chunks: list[dict] = []

    for src, doc in docs.items():
        digest = hashes[src]
        prev = prev_manifest.get(src)
        if not rebuild and prev and prev.get("sha256") == digest and prev.get("chunks"):
            stats["skipped"].append(src)
            kept_chunks.extend(c for c in prev_chunks if c.get("source") == src)
            continue

        try:
            vector_store.delete_source(col, src)      # 先清旧，避免版本残留
            chunks = splitter.split_document(doc)
            if not chunks:
                stats["errors"].append(f"{src}：切分后无内容")
                continue
            vectors = ollama_client.embed_texts([c.text for c in chunks])
            if len(vectors) != len(chunks):
                raise RuntimeError(f"向量数 {len(vectors)} 与切片数 {len(chunks)} 不一致")
            vector_store.upsert_chunks(col, chunks, vectors)
            kept_chunks.extend(c.to_dict() for c in chunks)
            stats["chunks_added"] += len(chunks)

            prev_manifest[src] = {
                "sha256": digest,
                "chars": doc.char_count,
                "chunks": len(chunks),
                "file_type": doc.file_type,
                "ingested_at": datetime.now().isoformat(timespec="seconds"),
            }
            stats["updated" if prev else "new"].append(src)
        except Exception as e:  # noqa: BLE001
            stats["errors"].append(f"{src}：{type(e).__name__} {e}")

    # 目录里已删除的文件，同步从库中移除
    for src in list(prev_manifest.keys()):
        if src not in docs:
            vector_store.delete_source(col, src)
            prev_manifest.pop(src)
            stats["removed"].append(src)

    kept_chunks.sort(key=lambda c: (c["source"], int(c["page"]), int(c["chunk_index"])))
    save_json(CHUNKS_JSON, kept_chunks)
    save_json(MANIFEST_JSON, prev_manifest)

    stats["chunks_in_file"] = len(kept_chunks)
    stats["chunks_in_chroma"] = vector_store.count(col)
    stats["consistent"] = stats["chunks_in_file"] == stats["chunks_in_chroma"]
    stats["manifest_files"] = len(prev_manifest)
    return stats


def print_summary(stats: dict) -> None:
    print("\n=== 入库结果 ===")
    print(f"文档目录      : {stats['docs_dir']}")
    print(f"有内容文件    : {len(stats['new']) + len(stats['updated']) + len(stats['skipped'])}")
    print(f"  新增 {len(stats['new'])} / 更新 {len(stats['updated'])} / 跳过 {len(stats['skipped'])} / 移除 {len(stats['removed'])}")
    print(f"本次新增切片  : {stats['chunks_added']}")
    print(f"切片总数      : {stats['chunks_in_file']}（向量库 {stats['chunks_in_chroma']}）")
    print(f"一致性        : {'OK' if stats['consistent'] else '不一致，需要 --rebuild'}")
    if stats["empty_files"]:
        print(f"空文件(忽略)  : {', '.join(stats['empty_files'])}")
    if stats["failed"]:
        print(f"加载失败      : {', '.join(stats['failed'])}")
    if stats["errors"]:
        print("错误：")
        for e in stats["errors"]:
            print("  -", e)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="入库：加载 → 切片 → 向量化 → Chroma")
    parser.add_argument("--docs", default=str(DOCS_DIR), help="源文档目录")
    parser.add_argument("--rebuild", action="store_true", help="清空重建（换模型/改切片参数时用）")
    args = parser.parse_args()

    stats = ingest(args.docs, rebuild=args.rebuild)
    print_summary(stats)
    return 0 if stats["consistent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
