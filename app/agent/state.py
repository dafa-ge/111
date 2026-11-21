"""AgentState：图的状态定义。

steps 用 Annotated[list, operator.add] 累加，其余字段后写覆盖。
"""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class AgentState(TypedDict, total=False):
    question: str
    session_id: str
    user_id: str
    history: list[dict]          # 短期记忆最近几轮
    memories: list[dict]         # 长期记忆（岗位等），用于填工具参数
    decision: dict               # {"kind": "tool"|"rag"|"answer", ...}
    route: str                   # tool / rag / answer / reject / compose
    tool_calls: list[dict]       # [{"name","args"}]
    tool_results: list[dict]     # 工具返回
    rag_hits: list[dict]         # 检索命中
    need_clarify: str            # 工具要求补充信息时的反问
    answer: str
    steps: Annotated[list, operator.add]   # 轨迹，面板要显示
