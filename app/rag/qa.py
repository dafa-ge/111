"""问答入口：检索 → 置信判断 → LLM 生成 → 可选打印召回片段。"""
from __future__ import annotations

import argparse
import sys
import time

from . import ollama_client, prompt, service
from .config import MEMORY_LLM_FALLBACK, TOP_N_FINAL
from .retriever import Hit, is_confident
from ..guardrail import BLOCK_TEXT_INPUT, BLOCK_TEXT_OUTPUT, check_input, check_output
from ..logger import log_turn, new_trace_id
from ..memory.extractor import extract, should_llm_fallback, summarize_with_llm
from ..memory.long_memory import get_long_memory
from ..memory.short_memory import get_short_memory

NO_ANSWER = "知识库中未找到相关条款。建议补充对应制度文件，或换一种问法（例如带上岗位、金额、时间等关键词）。"


def print_hits(hits: list[Hit]) -> None:
    print("\n=== 召回片段（验证命中用）===")
    if not hits:
        print("（无召回）")
        return
    for i, h in enumerate(hits, start=1):
        tag = "[邻接]" if h.neighbor else ""
        page = f"第{h.page}页" if h.page else "无页码"
        print(f"[{i}]{tag} {h.source} {page} | cos={h.cos_sim:.3f} bm25={h.bm25:.2f} rrf={h.rrf:.4f}")
        if h.section:
            print(f"     章节: {h.section}")
        body = h.text.replace("\n", " ")
        print(f"     片段: {body[:140]}{'…' if len(body) > 140 else ''}")


def ask(
    question: str,
    top_n: int = TOP_N_FINAL,
    debug: bool = False,
    session_id: str = "cli-session",
    user_id: str = "default",
    use_memory: bool = True,
) -> dict:
    started = time.time()
    trace_id = new_trace_id()

    guard = check_input(question)
    if guard.hit:
        log_turn(question=question, answer=BLOCK_TEXT_INPUT, user_id=user_id, mode="rag",
                 session_id=session_id, blocked=True, stage="pre", hit_rule=guard.rule,
                 trace_id=trace_id, elapsed=time.time() - started)
        if debug:
            print(f"\n[护栏] 前置拦截：{guard.rule}（{guard.reason}）")
        return {
            "question": question,
            "answer": BLOCK_TEXT_INPUT,
            "hits": [],
            "confident": False,
            "memories": [],
            "facts": [],
            "blocked": True,
            "block_reason": guard.rule,
            "trace_id": trace_id,
        }

    short = get_short_memory()
    long_mem = get_long_memory()

    memories: list[dict] = []
    facts: list[dict] = []
    history: list[dict] | None = None
    if use_memory:
        try:
            memories = long_mem.recall(user_id, question)
        except Exception:  # noqa: BLE001  记忆坏了不能影响问答
            memories = []
        try:
            history = short.messages_for_llm(session_id)[-4:] or None
        except Exception:  # noqa: BLE001
            history = None

    retriever = service.get_retriever()
    hits = retriever.search(question, top_n=top_n)
    confident = is_confident(hits)

    if confident and hits:
        user_msg = prompt.build_user_message(question, hits, memories)
        try:
            answer = ollama_client.chat(prompt.SYSTEM_PROMPT, user_msg, history=history)
        except ollama_client.OllamaError as e:
            answer = f"（模型没有给出最终答案：{e}）"
    else:
        answer = NO_ANSWER

    guard_out = check_output(answer, question, suspect=guard.suspect)
    answer_raw = None
    if guard_out.hit:
        answer_raw, answer = answer, BLOCK_TEXT_OUTPUT

    if use_memory:
        try:
            short.append(session_id, "user", question)
            short.append(session_id, "assistant", answer)
            if not guard_out.hit:
                facts = extract(question)
                if not facts and MEMORY_LLM_FALLBACK and should_llm_fallback(question):
                    facts = summarize_with_llm(question)
                if facts:
                    long_mem.remember(user_id, session_id, facts)
        except Exception:  # noqa: BLE001
            pass

    log_turn(question=question, answer=answer, user_id=user_id, mode="rag",
             session_id=session_id, route="direct" if confident else "reject",
             hits=hits, memories=memories, blocked=guard_out.hit,
             stage=guard_out.stage, hit_rule=guard_out.rule,
             suspect=guard.suspect or guard_out.suspect, answer_raw=answer_raw,
             trace_id=trace_id, elapsed=time.time() - started)

    if debug:
        print_hits(hits)
        if memories:
            print("\n=== 本轮带上的用户信息 ===")
            for m in memories:
                print(f"  - {m['text']}（{m.get('created_at')}，cos={m.get('cos')}）")
        if facts:
            print("\n=== 写入长期记忆 ===")
            for f in facts:
                print(f"  + {f['text']}")
        if guard.suspect:
            print(f"\n[护栏] 问题命中可疑词：{guard.rule}（{guard_out.reason or '答案复核通过'}）")
        if guard_out.hit:
            print(f"\n[护栏] 后置拦截：{guard_out.rule}（{guard_out.reason}）")

    return {
        "question": question,
        "answer": answer,
        "hits": hits,
        "confident": confident,
        "memories": memories,
        "facts": facts,
        "blocked": guard_out.hit,
        "block_reason": guard_out.rule,
        "trace_id": trace_id,
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="知识库问答（只依据文档回答）")
    parser.add_argument("question", nargs="?", help="问题；不填则进入交互模式")
    parser.add_argument("--debug", action="store_true", help="打印召回的参考片段")
    parser.add_argument("--top-n", type=int, default=TOP_N_FINAL, help="交给模型的片段数")
    args = parser.parse_args()

    if not args.question:
        print("交互模式，输入问题回车；输入 exit 退出。")
        while True:
            try:
                q = input("\n问> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q or q.lower() in {"exit", "quit"}:
                break
            res = ask(q, top_n=args.top_n, debug=args.debug)
            print("\n答案：", res["answer"])
            if res.get("blocked"):
                print(f"（已拦截：{res['block_reason']}）")
            elif not res["confident"]:
                print("（置信度不足，已拒答）")
        return 0

    res = ask(args.question, top_n=args.top_n, debug=args.debug)
    print("\n答案：", res["answer"])
    if res.get("blocked"):
        print(f"（已拦截：{res['block_reason']}）")
    elif not res["confident"]:
        print("（置信度不足，已拒答）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
