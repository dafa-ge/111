"""切片器：中文友好的递归切分（等价于 RecursiveCharacterTextSplitter 的逻辑，自实现以便讲清原理）。

切分顺序：段落 → 换行 → 句号/分号/叹号/问号 → 逗号 → 顿号 → 空格 → 硬切。
合并规则：相邻片段拼到接近 chunk_size；下一块开头带上一块尾部 overlap 个字符。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import CHUNK_OVERLAP, CHUNK_SIZE, MIN_CHUNK_LEN, SEPARATORS
from .loader import RawDoc


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str
    file_type: str
    page: int
    chunk_index: int          # 同文件内序号，从 0 开始（邻接扩展依赖它）
    section: str = ""

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source": self.source,
            "file_type": self.file_type,
            "page": self.page,
            "chunk_index": self.chunk_index,
            "section": self.section,
        }


@dataclass
class SplitReport:
    chunks: list[Chunk] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def _split_recursive(text: str, seps: list[str], level: int, size: int) -> list[str]:
    """按分隔符优先级递归切到 <= size 的片段，尽量保留分隔符。"""
    if len(text) <= size:
        return [text]
    if level >= len(seps):
        return [text[i : i + size] for i in range(0, len(text), size)]

    sep = seps[level]
    raw_parts = text.split(sep)
    pieces: list[str] = []
    for i, part in enumerate(raw_parts):
        if i < len(raw_parts) - 1:
            part = part + sep        # 分隔符留在片段尾部，减少语义断裂
        if part:
            pieces.append(part)

    out: list[str] = []
    for p in pieces:
        if len(p) <= size:
            out.append(p)
        else:
            out.extend(_split_recursive(p, seps, level + 1, size))
    return out


def _merge(pieces: list[str], size: int, overlap: int) -> list[str]:
    """把碎片拼成接近 size 的块，块间保留 overlap 上下文。"""
    merged: list[str] = []
    cur = ""
    for p in pieces:
        if not p:
            continue
        if len(cur) + len(p) <= size:
            cur += p
            continue
        if cur.strip():
            merged.append(cur)
        cur = (cur[-overlap:] + p) if (overlap and cur) else p
    if cur.strip():
        merged.append(cur)
    return merged


_CODE_LINE = re.compile(r"^\s*(?:[一二三四五六七八九十]+、|第[一二三四五六七八九十]+[章条节]|\d+[\.、])")


def _guess_section(text: str) -> str:
    for line in text.split("\n")[:3]:
        line = line.strip()
        if line and _CODE_LINE.match(line):
            return line[:30]
    return ""


def split_document(
    doc: RawDoc,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    separators: list[str] | None = None,
) -> list[Chunk]:
    seps = separators or SEPARATORS
    chunks: list[Chunk] = []
    idx = 0

    for page, text in doc.segments:
        pieces = _split_recursive(text, seps, 0, chunk_size)
        for seg in _merge(pieces, chunk_size, overlap):
            seg = seg.strip()
            if not seg:
                continue
            if len(seg) < MIN_CHUNK_LEN and chunks:
                # 过短的碎片并入上一块，避免产生"半句话"独立成块
                chunks[-1].text = f"{chunks[-1].text}\n{seg}"
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.source}::{page}::{idx:03d}",
                    text=seg,
                    source=doc.source,
                    file_type=doc.file_type,
                    page=page,
                    chunk_index=idx,
                    section=_guess_section(seg),
                )
            )
            idx += 1
    return chunks


def split_report(chunks: list[Chunk]) -> dict:
    if not chunks:
        return {"count": 0, "avg": 0, "min": 0, "max": 0, "fragments": 0, "frag_rate": 0.0}
    lens = [len(c.text) for c in chunks]
    frags = sum(1 for n in lens if n < 100)
    return {
        "count": len(lens),
        "avg": round(sum(lens) / len(lens), 1),
        "min": min(lens),
        "max": max(lens),
        "fragments": frags,                        # <100 字的碎片块
        "frag_rate": round(frags / len(lens), 3),
    }
