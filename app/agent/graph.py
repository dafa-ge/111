"""LangGraph 状态机：decide -> (act | rag | direct | reject) -> compose/clarify/END。

图的形状（与 react.py 的手工编排完全等价）：

    decide ──tool──▶ act ──有反问──▶ clarify ─▶ END
       │              └──无反问──▶ compose ─▶ END
       ├──rag──▶ rag ──置信──▶ compose ─▶ END
       │            └──不置信──▶ reject ─▶ END
       ├──answer──▶ direct ─▶ END
       └──其它──▶ reject ─▶ END

compose 之后没有回边，从结构上杜绝"调完工具再想再调"的死循环。
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from . import react
from .state import AgentState


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("decide", react.node_decide)
    g.add_node("act", react.node_act)
    g.add_node("rag", react.node_rag)
    g.add_node("compose", react.node_compose)
    g.add_node("direct", react.node_direct)
    g.add_node("clarify", react.node_clarify)
    g.add_node("reject", react.node_reject)

    g.set_entry_point("decide")

    g.add_conditional_edges("decide", react.route_after_decide, {
        "act": "act",
        "rag": "rag",
        "direct": "direct",
        "reject": "reject",
    })
    g.add_conditional_edges("act", react.route_after_act, {
        "compose": "compose",
        "clarify": "clarify",
    })
    g.add_conditional_edges("rag", react.route_after_rag, {
        "compose": "compose",
        "reject": "reject",
    })

    g.add_edge("compose", END)
    g.add_edge("direct", END)
    g.add_edge("clarify", END)
    g.add_edge("reject", END)

    return g.compile()


GRAPH = build_graph()


def run(question: str, session_id: str = "agent-default", user_id: str = "default",
        memories: list[dict] | None = None, history: list[dict] | None = None) -> dict:
    """跑一遍图，返回最终状态（含 steps 轨迹）。"""
    init: dict = {
        "question": question,
        "session_id": session_id,
        "user_id": user_id,
        "memories": memories,
        "history": history,
        "steps": [],
    }
    config = {"recursion_limit": 8}      # 硬顶：这套图最多 4 步，8 是保险
    return GRAPH.invoke(init, config=config)
