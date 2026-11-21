"""文档加载器：txt / docx / pdf -> 带来源与页码的纯文本。
原则：任何一份文件读不出内容，都要显式报出来，绝不静默跳过。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import DOCS_DIR, SUPPORTED_EXTS


@dataclass
class RawDoc:
    source: str                     # 文件名，作为来源标识（入库后不要改名）
    file_type: str                  # txt / docx / pdf
    path: str
    segments: list[tuple[int, str]] = field(default_factory=list)   # [(页码或0, 文本)]
    char_count: int = 0

    @property
    def text(self) -> str:
        return "\n\n".join(t for _, t in self.segments)


@dataclass
class LoadReport:
    docs: list[RawDoc] = field(default_factory=list)
    empty_files: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)      # "文件名：原因"
    ignored: list[str] = field(default_factory=list)


def clean_text(text: str) -> str:
    """统一空白、去 BOM、去装饰线、压缩空行。不做软换行合并（会破坏条款编号）。"""
    if not text:
        return ""
    text = text.replace("\ufeff", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\u3000]+", " ", text)
    text = re.sub(r"\n(?:[ \t]*\n){2,}", "\n\n", text)
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln and not re.fullmatch(r"[-=_*—·]{3,}", ln)]
    return "\n".join(lines).strip()


def _read_txt(path: Path) -> list[tuple[int, str]]:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return [(0, raw.decode(enc))]
        except UnicodeDecodeError:
            continue
    return [(0, raw.decode("utf-8", errors="replace"))]


def _read_docx(path: Path) -> list[tuple[int, str]]:
    from docx import Document  # 延迟导入，没装也不影响 txt/pdf

    doc = Document(str(path))
    parts: list[str] = []
    for p in doc.paragraphs:
        if p.text.strip():
            parts.append(p.text.strip())
    # 表格必须读：住宿标准、报销额度通常放在表格里
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip().replace("\n", " ") for c in row.cells]
            cells = [c for c in cells if c]
            if not cells:
                continue
            parts.append("：".join(cells) if len(cells) == 2 else " | ".join(cells))
    return [(0, "\n".join(parts))]


def _read_pdf(path: Path) -> list[tuple[int, str]]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    segments: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            txt = page.extract_text() or ""
        except Exception:  # noqa: BLE001
            txt = ""
        segments.append((i, txt))
    return segments


READERS = {".txt": _read_txt, ".md": _read_txt, ".docx": _read_docx, ".pdf": _read_pdf}


def load_file(path: Path) -> RawDoc:
    reader = READERS[path.suffix.lower()]
    segments = [(p, clean_text(t)) for p, t in reader(path)]
    segments = [(p, t) for p, t in segments if t]
    return RawDoc(
        source=path.name,
        file_type=path.suffix.lower().lstrip("."),
        path=str(path),
        segments=segments,
        char_count=sum(len(t) for _, t in segments),
    )


def load_directory(docs_dir: Path | str = DOCS_DIR) -> LoadReport:
    docs_dir = Path(docs_dir)
    report = LoadReport()
    if not docs_dir.exists():
        report.failed.append(f"目录不存在：{docs_dir}")
        return report

    for path in sorted(docs_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTS:
            report.ignored.append(path.name)
            continue
        try:
            doc = load_file(path)
        except Exception as e:  # noqa: BLE001
            report.failed.append(f"{path.name}：{type(e).__name__} {e}")
            continue
        if doc.char_count == 0:
            # 0 字节 / 扫描版 PDF / 空文档，一律显式报出
            report.empty_files.append(path.name)
            continue
        report.docs.append(doc)
    return report
