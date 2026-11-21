"""回归评测：把问题集跑一遍检索，算 recall@4、来源命中率、拒答准确率。
只测检索层（不调 LLM），因为检索错了，LLM 再强也只是编得更像。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime

from . import service
from .config import QA_PAIRS_JSON, REPORTS_DIR, TOP_N_FINAL
from .retriever import is_confident


def load_pairs(path=QA_PAIRS_JSON) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate(top_n: int = TOP_N_FINAL) -> dict:
    retriever = service.get_retriever()
    pairs = load_pairs()
    rows: list[dict] = []

    for p in pairs:
        hits = retriever.search(p["q"], top_n=top_n)
        confident = is_confident(hits)
        if p.get("should_refuse"):
            ok = not confident
            reason = "正确拒答" if ok else f"误命中：{hits[0].source}（cos={hits[0].cos_sim:.3f}）"
            source_hit = False
            kw_hit = False
        else:
            src_hit = any(p["expect_source"] in h.source for h in hits)
            kw_hit = any(
                any(k in h.text for k in p.get("expect_keywords", [])) for h in hits
            )
            ok = src_hit and kw_hit
            missing = []
            if not src_hit:
                missing.append("来源未命中")
            if not kw_hit:
                missing.append(f"关键词缺失({p.get('expect_keywords')})")
            reason = "命中" if ok else "；".join(missing)
            source_hit = src_hit

        rows.append(
            {
                "q": p["q"],
                "ok": ok,
                "reason": reason,
                "should_refuse": bool(p.get("should_refuse")),
                "top": [
                    {"source": h.source, "cos": round(h.cos_sim, 3), "bm25": round(h.bm25, 2)}
                    for h in hits[:3]
                ],
            }
        )

    answerable = [r for r in rows if not r["should_refuse"]]
    refusable = [r for r in rows if r["should_refuse"]]
    recall = round(sum(1 for r in answerable if r["ok"]) / len(answerable), 3) if answerable else 0.0
    refuse_acc = round(sum(1 for r in refusable if r["ok"]) / len(refusable), 3) if refusable else 0.0

    result = {
        "rows": rows,
        "recall@4": recall,
        "refuse_accuracy": refuse_acc,
        "total": len(rows),
        "passed": sum(1 for r in rows if r["ok"]),
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    _write_report(result)
    return result


def _write_report(result: dict) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    lines = [
        "# 回归评测",
        "",
        f"- 时间：{result['time']}",
        f"- recall@4：{result['recall@4']}",
        f"- 拒答准确率：{result['refuse_accuracy']}",
        f"- 通过：{result['passed']}/{result['total']}",
        "",
        "| # | 问题 | 结果 | 说明 | top1 |",
        "|---|---|---|---|---|",
    ]
    for i, r in enumerate(result["rows"], start=1):
        top1 = r["top"][0]["source"] if r["top"] else "-"
        lines.append(
            f"| {i} | {r['q']} | {'PASS' if r['ok'] else 'FAIL'} | {r['reason']} | {top1} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result["report_path"] = str(path)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="回归评测检索效果")
    parser.add_argument("--top-n", type=int, default=TOP_N_FINAL)
    args = parser.parse_args()

    res = evaluate(args.top_n)
    for r in res["rows"]:
        print(f"{'PASS' if r['ok'] else 'FAIL'}  {r['q']}  —— {r['reason']}")
    print(f"\nrecall@4 = {res['recall@4']}　拒答准确率 = {res['refuse_accuracy']}　通过 {res['passed']}/{res['total']}")
    print(f"报告：{res['report_path']}")
    return 0 if res["passed"] == res["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
