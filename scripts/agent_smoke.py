"""ReAct 智能体验收脚本。

用法：
  python scripts\agent_smoke.py 3          # 只跑第 3 个用例
  python scripts\agent_smoke.py all        # 全部跑（很久，>30 分钟）
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.graph import run  # noqa: E402

MEM_MARKET = [{"kind": "岗位", "text": "岗位：市场岗"}]

CASES: dict[str, tuple[str, list | None]] = {
    "1": ("你好", None),
    "2": ("事假怎么算", None),
    "3": ("我还有几天年假", None),
    "4a": ("我出差住宿能报多少", MEM_MARKET),          # 没说省内省外 → 应反问
    "4b": ("省内出差住宿能报多少", MEM_MARKET),        # 应 350
    "5": ("食堂几点开饭", None),
    "6": ("我工龄几年，年假还剩多少", None),
}


def show(title: str, state: dict, seconds: float) -> None:
    print(f"\n=== {title} ===")
    print("路由：", state.get("route"))
    for s in state.get("steps", []):
        print(f"  [{s['node']}] {s['detail']}    {s['seconds']}s")
    if state.get("tool_results"):
        print("工具返回：", json.dumps(state["tool_results"], ensure_ascii=False)[:400])
    print("答案：", (state.get("answer") or "").strip())
    print(f"总耗时：{round(seconds, 1)}s")


def main() -> None:
    args = sys.argv[1:] or ["3"]
    which = args[0].lower()
    keys = list(CASES) if which == "all" else [a.lower() for a in args]
    for k in keys:
        if k not in CASES:
            print(f"没有用例 {k}，可选：{', '.join(CASES)}")
            continue
        q, mem = CASES[k]
        t0 = time.time()
        try:
            state = run(q, session_id=f"agent-{k}", memories=mem)
        except Exception as e:  # noqa: BLE001
            print(f"\n=== 用例 {k} 出错 ===\n{type(e).__name__}: {e}")
            continue
        show(f"用例 {k}｜{q}", state, time.time() - t0)


if __name__ == "__main__":
    main()
