"""ReAct 的节点实现。

节点函数与 LangGraph 解耦：都是 (state) -> 增量 dict，既可以挂到 StateGraph，
也可以被 run_agent() 手工顺序调用（单测 / LangGraph 出问题时的降级路径）。
"""
from __future__ import annotations

import json
import time

from ..rag import prompt, service
from ..rag.qa import NO_ANSWER
from ..rag.retriever import is_confident
from . import llm
from .tools import call_tool

POSTS = ("职能岗", "技术岗", "市场岗")


def _step(node: str, detail: str, seconds: float, **extra) -> dict:
    return {"node": node, "detail": detail, "seconds": round(seconds, 1), **extra}


def post_from_memories(memories: list[dict] | None) -> str | None:
    """从长期记忆里挖岗位，用来补工具参数。"""
    for m in memories or []:
        if m.get("kind") == "岗位":
            text = m.get("text") or ""
            for p in POSTS:
                if p in text:
                    return p
    return None


def hits_to_text(hits) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        blocks.append(f"[{i}] {h.source}\n{h.text}")
    return "【参考资料】\n" + "\n\n".join(blocks)


def tools_to_text(results: list[dict]) -> str:
    lines = []
    for r in results:
        body = json.dumps(r.get("result") or {}, ensure_ascii=False)
        lines.append(f"- {r['name']}({json.dumps(r.get('args') or {}, ensure_ascii=False)}) → {body}")
    return "【工具结果】\n" + "\n".join(lines)


# ---------------------------------------------------------------- 节点
def node_decide(state: dict) -> dict:
    t0 = time.time()
    memories = state.get("memories")
    if memories is None:
        from ..memory.long_memory import get_long_memory
        memories = get_long_memory().recall(state.get("user_id") or "default",
                                             state["question"])
    dec = llm.decide(state["question"], memories, state.get("history"))
    if dec.get("kind") == "tool":
        detail = "调工具 " + "、".join(
            f"{c['name']}({json.dumps(c.get('args') or {}, ensure_ascii=False)})"
            for c in dec.get("calls", []))
    elif dec.get("kind") == "rag":
        detail = "需要查文档"
    else:
        detail = "直接回答"
    return {
        "decision": dec,
        "memories": memories,
        "route": dec.get("kind") or "reject",
        "steps": [_step("决定", detail, time.time() - t0)],
    }


def node_act(state: dict) -> dict:
    t0 = time.time()
    results: list[dict] = []
    clarify = ""
    for c in (state.get("decision") or {}).get("calls", []):
        args = dict(c.get("args") or {})
        if c.get("name") == "check_travel_standard" and not args.get("post"):
            post = post_from_memories(state.get("memories"))
            if post:
                args["post"] = post
        if c.get("name") == "query_annual_leave":
            # 身份以登录态为准，不用模型传上来的 user_id（否则 U002 会查到 U001 的年假）
            args["user_id"] = state.get("user_id") or "default"
        r = call_tool(c.get("name"), args)
        results.append({"name": c.get("name"), "args": args, "result": r})
        if r.get("need_clarify"):
            clarify = r.get("question") or ""

    names = "、".join(r["name"] for r in results) or "无"
    return {
        "tool_results": results,
        "need_clarify": clarify,
        "steps": [_step("调工具", f"{names} 已返回", time.time() - t0)],
    }


def node_rag(state: dict) -> dict:
    t0 = time.time()
    hits = service.get_retriever().search(state["question"])
    confident = is_confident(hits)
    step = _step("查文档",
                 f"召回 {len(hits)} 段，{'可信' if confident else '置信不足，直接拒答'}",
                 time.time() - t0)
    if not confident:
        return {"rag_hits": [], "route": "reject", "answer": NO_ANSWER, "steps": [step]}
    return {"rag_hits": hits, "route": "compose", "steps": [step]}


def node_compose(state: dict) -> dict:
    t0 = time.time()
    if state.get("tool_results"):
        materials = tools_to_text(state["tool_results"])
    else:
        materials = hits_to_text(state.get("rag_hits") or [])
    answer = llm.compose(state["question"], materials, state.get("memories"))
    return {"answer": answer, "steps": [_step("组织答案", f"{len(answer)} 字", time.time() - t0)]}


def node_direct(state: dict) -> dict:
    ans = (state.get("decision") or {}).get("answer") or NO_ANSWER
    return {"answer": ans, "steps": [_step("直接回答", "无需检索/工具", 0.0)]}


def node_clarify(state: dict) -> dict:
    q = state.get("need_clarify") or "需要补充信息才能查"
    return {"answer": q, "steps": [_step("反问", q, 0.0)]}


def node_reject(state: dict) -> dict:
    return {"answer": state.get("answer") or NO_ANSWER,
            "steps": [_step("拒答", "没有可靠依据", 0.0)]}


# ---------------------------------------------------------------- 条件边
def route_after_decide(state: dict) -> str:
    r = state.get("route") or "reject"
    if r == "tool":
        return "act"
    if r == "rag":
        return "rag"
    if r == "answer":
        return "direct"
    return "reject"


def route_after_act(state: dict) -> str:
    return "clarify" if state.get("need_clarify") else "compose"


def route_after_rag(state: dict) -> str:
    return "compose" if state.get("route") == "compose" else "reject"


# ---------------------------------------------------------------- 手工编排（降级 / 单测）
def run_agent(question: str, session_id: str = "agent-default",
              user_id: str = "default", memories: list[dict] | None = None,
              history: list[dict] | None = None) -> dict:
    """不依赖 LangGraph 的等价执行，行为与图一致。"""
    state: dict = {"question": question, "session_id": session_id,
                   "user_id": user_id, "memories": memories, "history": history,
                   "steps": []}
    state.update(node_decide(state))
    r = route_after_decide(state)
    if r == "act":
        state.update(node_act(state))
        if route_after_act(state) == "clarify":
            state.update(node_clarify(state))
        else:
            state.update(node_compose(state))
    elif r == "rag":
        state.update(node_rag(state))
        if route_after_rag(state) == "compose":
            state.update(node_compose(state))
    elif r == "direct":
        state.update(node_direct(state))
    else:
        state.update(node_reject(state))
    return state
