"""文档体检：入库前先确认"每份文件到底有没有内容、关键条款在不在"。
空文件、扫描件、乱码在这里暴露，避免污染后面的向量库。
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from .config import DOCS_DIR, REPORTS_DIR, SUPPORTED_EXTS

PROBE_KEYWORDS = [
    "住宿标准", "住宿上限", "报销", "市场岗", "技术岗", "职能岗",
    "保密", "考勤", "休假", "社保", "公积金", "离职", "试用期", "AI",
]


def _locate(path: Path, keywords: list[str]) -> dict[str, list[int]]:
    """关键词 → 命中行号（从 1 开始）。非文本格式返回空。"""
    hits: dict[str, list[int]] = {}
    if path.suffix.lower() in {".txt", ".md"}:
        for enc in ("utf-8-sig", "utf-8", "gb18030"):
            try:
                text = path.read_text(encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            text = path.read_text(encoding="utf-8", errors="replace")
        for i, line in enumerate(text.splitlines(), start=1):
            for kw in keywords:
                if kw in line:
                    hits.setdefault(kw, []).append(i)
    return hits


def run(docs_dir: Path | str | None = None) -> dict:
    docs_dir = Path(docs_dir or DOCS_DIR)
    result: dict = {
        "dir": str(docs_dir),
        "files": [],
        "empty": [],
        "failed": [],
        "keyword_map": {},
        "ok": True,
    }

    if not docs_dir.exists():
        result["ok"] = False
        result["failed"].append(f"目录不存在：{docs_dir}")
        return result

    for path in sorted(docs_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTS:
            continue
        row: dict = {
            "name": path.name,
            "type": path.suffix.lower().lstrip("."),
            "bytes": path.stat().st_size,
        }
        chars = 0
        pages = 0
        status = "正常"
        try:
            if path.suffix.lower() == ".pdf":
                from pypdf import PdfReader
                reader = PdfReader(str(path))
                pages = len(reader.pages)
        except Exception:  # noqa: BLE001
            status = "读取失败"

        try:
            kw_hits = _locate(path, PROBE_KEYWORDS)
        except Exception:  # noqa: BLE001
            kw_hits = {}

        if path.suffix.lower() in {".txt", ".md"}:
            for enc in ("utf-8-sig", "utf-8", "gb18030"):
                try:
                    chars = len(path.read_text(encoding=enc))
                    break
                except UnicodeDecodeError:
                    continue

        if path.stat().st_size == 0 or (path.suffix.lower() in {".txt", ".md"} and chars == 0):
            status = "空文件"
            result["empty"].append(path.name)
        if status == "读取失败":
            result["failed"].append(path.name)

        row.update({"chars": chars, "pages": pages, "status": status, "keywords": kw_hits})
        result["files"].append(row)

        for kw, lines in kw_hits.items():
            result["keyword_map"].setdefault(kw, []).append(f"{path.name}:{'/'.join(map(str, lines[:3]))}")

    result["ok"] = not result["empty"] and not result["failed"]
    _write_report(result)
    return result


def _write_report(result: dict) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = Path(REPORTS_DIR) / "selfcheck.md"

    lines = [
        "# 文档体检报告",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 目录：`{result['dir']}`",
        f"- 结论：{'通过' if result['ok'] else '有问题，需先处理'}",
        "",
        "## 逐份检查",
        "",
        "| 文件 | 格式 | 字节 | 字符数 | 页数 | 状态 |",
        "|---|---|---|---|---|---|",
    ]
    for f in result["files"]:
        lines.append(
            f"| {f['name']} | {f['type']} | {f['bytes']} | {f['chars']} | {f['pages'] or '-'} | {f['status']} |"
        )

    lines += ["", "## 关键词定位", ""]
    if result["keyword_map"]:
        lines += ["| 关键词 | 命中位置（文件:行号） |", "|---|---|"]
        for kw, locs in sorted(result["keyword_map"].items()):
            lines.append(f"| {kw} | {'；'.join(locs)} |")
    else:
        lines.append("（未命中任何探测关键词）")

    if result["empty"]:
        lines += ["", "## 空文件", "", "以下文件 0 字节或无文本内容，**不会入库**，需补充内容后重跑："]
        lines += [f"- {n}" for n in result["empty"]]
    if result["failed"]:
        lines += ["", "## 读取失败", ""]
        lines += [f"- {n}" for n in result["failed"]]

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result["report_path"] = str(report_path)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    res = run()
    print(f"目录: {res['dir']}")
    for f in res["files"]:
        print(f"  {f['name']:<30} {f['chars']:>6} 字  {f['status']}")
    if res["empty"]:
        print(f"\n空文件 {len(res['empty'])} 份：{', '.join(res['empty'])}")
    print(f"\n报告: {res.get('report_path')}")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
