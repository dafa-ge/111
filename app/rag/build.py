"""一键建库：体检 → 加载 → 切片 → 向量化 → 入库 → 自检检索 → 出报告。
这是 L1 自动化的入口，也是 L2 增量同步的入口（每次运行先比对指纹）。
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime

from . import ingest, ollama_client, selfcheck, service
from .config import (
    CHUNK_OVERLAP, CHUNK_SIZE, DOCS_DIR, EMBED_MODEL, LLM_MODEL,
    OLLAMA_BASE, REPORTS_DIR,
)
from .retriever import is_confident

SELF_TESTS = [
    {
        "question": "辰星科技市场岗省内出差住宿标准是多少？",
        "expect_source": "费用报销",
        "expect_kw": "350",
        "label": "主用例：岗位差旅标准",
    },
    {
        "question": "报销金额超过10000元需要谁审批？",
        "expect_source": "费用报销",
        "expect_kw": "总经理",
        "label": "审批流程（数字+角色）",
    },
    {
        "question": "食堂几点开饭？",
        "refuse": True,
        "label": "幻觉测试：文档无此内容，应拒答",
    },
]


def chunk_stats(chunks: list[dict]) -> dict:
    if not chunks:
        return {"count": 0, "avg": 0, "min": 0, "max": 0, "frag_rate": 0.0}
    lens = [len(c["text"]) for c in chunks]
    frags = sum(1 for n in lens if n < 100)
    return {
        "count": len(lens),
        "avg": round(sum(lens) / len(lens), 1),
        "min": min(lens),
        "max": max(lens),
        "frag_rate": round(frags / len(lens), 3),
    }


def run_self_tests() -> list[dict]:
    retriever = service.get_retriever()
    results = []
    for case in SELF_TESTS:
        hits = retriever.search(case["question"])
        confident = is_confident(hits)
        if case.get("refuse"):
            passed = not confident
            detail = "已拒答（无相关召回）" if passed else f"误答风险：最高 cos={hits[0].cos_sim:.3f}"
        else:
            top = hits[:3]
            passed = any(
                case["expect_source"] in h.source and case["expect_kw"] in h.text
                for h in top
            )
            detail = "; ".join(
                f"{h.source}#{h.chunk_id.rsplit('::', 1)[-1]} cos={h.cos_sim:.3f}" for h in top[:3]
            )
        results.append(
            {
                "label": case["label"],
                "question": case["question"],
                "passed": passed,
                "confident": confident,
                "detail": detail,
                "top": [
                    {
                        "source": h.source,
                        "page": h.page,
                        "cos": round(h.cos_sim, 3),
                        "bm25": round(h.bm25, 2),
                        "neighbor": h.neighbor,
                        "head": h.head(80),
                    }
                    for h in hits[:4]
                ],
            }
        )
    return results


def write_report(health: dict, check: dict, ing: dict, stats: dict, tests: list[dict]) -> str:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now()
    path = REPORTS_DIR / f"build_{ts.strftime('%Y%m%d_%H%M%S')}.md"

    passed = sum(1 for t in tests if t["passed"])
    lines = [
        "# 建库报告",
        "",
        f"- 时间：{ts.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 文档目录：`{ing['docs_dir']}`",
        "",
        "## 1 环境",
        "",
        f"- Ollama：`{OLLAMA_BASE}` {'可用' if health['ok'] else '不可用：' + str(health['error'])}",
        f"- 向量模型：`{EMBED_MODEL}`　对话模型：`{LLM_MODEL}`",
        f"- 切片参数：chunk_size={CHUNK_SIZE}，overlap={CHUNK_OVERLAP}",
        "",
        "## 2 文档体检",
        "",
        "| 文件 | 格式 | 字符数 | 状态 |",
        "|---|---|---|---|",
    ]
    for f in check["files"]:
        lines.append(f"| {f['name']} | {f['type']} | {f['chars']} | {f['status']} |")

    if check["empty"]:
        lines += ["", f"**空文件（未入库）**：{'、'.join(check['empty'])}"]

    lines += [
        "",
        "## 3 入库结果",
        "",
        f"- 新增 {len(ing['new'])} / 更新 {len(ing['updated'])} / 跳过 {len(ing['skipped'])} / 移除 {len(ing['removed'])}",
        f"- 切片总数：{ing['chunks_in_file']}　向量库：{ing['chunks_in_chroma']}　一致性：{'OK' if ing['consistent'] else '不一致'}",
        f"- 切片统计：平均 {stats['avg']} 字，最短 {stats['min']}，最长 {stats['max']}，碎片率 {stats['frag_rate']}",
    ]
    if ing["errors"]:
        lines += ["", "**错误**："] + [f"- {e}" for e in ing["errors"]]

    lines += ["", "## 4 检索自检", "", f"通过 {passed}/{len(tests)}", ""]
    for t in tests:
        mark = "PASS" if t["passed"] else "FAIL"
        lines.append(f"### [{mark}] {t['label']}")
        lines.append(f"- 问题：{t['question']}")
        lines.append(f"- 判断：{t['detail']}")
        for i, h in enumerate(t["top"], start=1):
            tag = "（邻接）" if h["neighbor"] else ""
            lines.append(
                f"  {i}. {h['source']} 第{h['page'] or '-'}页{tag} cos={h['cos']} bm25={h['bm25']} | {h['head']}"
            )
        lines.append("")

    lines += [
        "## 5 结论",
        "",
        f"- 文档：{len(check['files'])} 份，可用 {len(check['files']) - len(check['empty'])} 份",
        f"- 入库一致性：{'通过' if ing['consistent'] else '不通过'}",
        f"- 检索自检：{passed}/{len(tests)}",
        f"- 总判定：{'可以进入问答验收' if ing['consistent'] and passed == len(tests) else '需处理未通过项'}",
    ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="一键建库（L1）")
    parser.add_argument("--docs", default=str(DOCS_DIR), help="源文档目录")
    parser.add_argument("--rebuild", action="store_true", help="清空重建")
    parser.add_argument("--skip-tests", action="store_true", help="跳过检索自检（离线时用）")
    args = parser.parse_args()

    health = ollama_client.health_check()
    print(f"[1/4] 环境检查：Ollama {'可用' if health['ok'] else '不可用'}")
    if not health["ok"]:
        print("      ", health["error"])
        return 2
    if EMBED_MODEL.split(":")[0] not in " ".join(health["models"]):
        print(f"      警告：模型 {EMBED_MODEL} 未在 ollama list 中出现")

    print("[2/4] 文档体检")
    check = selfcheck.run(args.docs)
    print(f"      文件 {len(check['files'])} 份，空文件 {len(check['empty'])} 份")

    print("[3/4] 入库")
    ing = ingest.ingest(args.docs, rebuild=args.rebuild)
    ingest.print_summary(ing)
    if not ing["consistent"]:
        print("      一致性校验失败，终止")
        return 1

    chunks = ingest.load_chunks()
    stats = chunk_stats(chunks)

    tests: list[dict] = []
    if not args.skip_tests:
        print("[4/4] 检索自检")
        tests = run_self_tests()
        for t in tests:
            print(f"      {'PASS' if t['passed'] else 'FAIL'}  {t['label']}")

    report = write_report(health, check, ing, stats, tests)
    print(f"\n报告已生成：{report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
