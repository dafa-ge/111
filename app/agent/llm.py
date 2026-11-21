"""ReAct 的两次 LLM 调用：决策（带工具 schema）与答案组织。"""
from __future__ import annotations

import json

from ..rag import config, prompt
from ..rag.ollama_client import _post
from .tools import TOOL_SCHEMAS

# 制度类关键词：模型没给路由标记时的兜底，避免它自己直接答制度题（幻觉入口）
RULE_KEYWORDS = (
    "请假", "事假", "病假", "年假", "报销", "发票", "审批", "试用期", "离职",
    "社保", "公积金", "竞业", "保密", "出差", "住宿", "餐补", "加班", "考勤",
    "薪酬", "工资", "调岗", "培训",
)

DECIDE_SYSTEM = """你是「辰星科技」的企业知识库助手。拿到问题后，第一件事是判断走哪条路。

可用的两个工具：
- query_annual_leave：查员工年假余额、已用、累计工龄。用户问“我还有几天年假”“我工龄几年”时用。
- check_travel_standard：查差旅住宿上限、餐补、交通标准。参数 post（岗位）、scope（省内/省外）。
  用户没说省内还是省外时，**scope 留空**，不要猜。

三条路径，只选一条：
1. 需要员工个人数据或差旅标准 → 直接调用工具，不要输出任何路由标记。
2. 需要公司制度条款（请假、报销、审批、试用期、社保、公积金、竞业、保密、考勤等）→
   只输出一行 `【路由】rag`，不要作答，系统会去检索文档。
3. 闲聊、打招呼、你能确定直接回答的常识 → 第一行输出 `【路由】direct`，第二行起给答案（不超过 80 字）。

不要编造制度条款、金额、天数。拿不准就走 rag。

{memory_block}"""

COMPOSE_SYSTEM = """你是「辰星科技」的企业知识库助手。只依据下面给出的【工具结果】或【参考资料】回答。

规则：
1. 数字、金额、天数、期限必须与工具结果或参考资料完全一致，不得改写、不得估算。
2. 材料里没有的内容一律不补充，宁可说"资料里没有提到"。
3. 答案控制在 150 字以内，先给结论，再列适用条件。
4. 最后一行用「【答案】」标记开始正文（这是给系统解析用的，不是给用户看的），形如：
【答案】市场岗省内出差住宿上限 350 元/晚。

{memory_block}"""


def _messages(system: str, user: str, history: list[dict] | None = None) -> list[dict]:
    msgs = [{"role": "system", "content": system}]
    if history:
        msgs.extend(history)
    msgs.append({"role": "user", "content": user})
    return msgs


def decide(question: str, memories: list[dict] | None = None,
           history: list[dict] | None = None, timeout: int = 1200) -> dict:
    """第 1 次推理：让模型在「调工具 / 查文档 / 直接答」之间选一条。

    返回 {"kind": "tool"|"rag"|"answer"|"reject", ...}
    """
    mem_block = prompt.build_memory_block(memories or [])
    system = DECIDE_SYSTEM.replace("{memory_block}", mem_block).strip()
    payload = {
        "model": config.LLM_MODEL,
        "messages": _messages(system, question, history),
        "tools": TOOL_SCHEMAS,
        "stream": False,
        "options": {
            "temperature": 0,
            "num_ctx": config.LLM_NUM_CTX,
            "num_predict": config.LLM_NUM_PREDICT,
        },
    }
    resp = _post("/api/chat", payload, timeout=timeout)
    return parse_decision(resp.get("message") or {}, question)


def parse_decision(msg: dict, question: str = "") -> dict:
    """把模型输出解析成决策。原生 tool_calls 优先，其次【路由】标记，最后关键词兜底。"""
    calls = msg.get("tool_calls") or []
    if calls:
        out = []
        for c in calls[:3]:
            fn = c.get("function") or {}
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:  # noqa: BLE001
                    args = {}
            if fn.get("name"):
                out.append({"name": fn["name"], "args": args or {}})
        if out:
            return {"kind": "tool", "calls": out}

    text = (msg.get("content") or "").strip()
    idx = text.rfind("【路由】")          # 模型可能在思考里演练，取最后一个
    if idx >= 0:
        tail = text[idx + len("【路由】"):].lstrip("：: \n").strip()
        low = tail[:12].lower()
        if low.startswith("rag"):
            return {"kind": "rag"}
        if low.startswith("direct"):
            body = tail[len("direct"):].lstrip("：: \n").strip()
            return {"kind": "answer", "answer": body or text}

    # 没有标记：可能是直接答，也可能是漏标。含制度关键词就强制走文档。
    if any(k in question for k in RULE_KEYWORDS):
        return {"kind": "rag", "note": "无路由标记且含制度关键词，强制检索"}
    if text:
        return {"kind": "answer", "answer": text}
    return {"kind": "reject"}


def compose(question: str, materials: str, memories: list[dict] | None = None,
            timeout: int = 1200) -> str:
    """第 2 次推理：拿着工具结果或文档片段组织最终答案。"""
    mem_block = prompt.build_memory_block(memories or [])
    system = COMPOSE_SYSTEM.replace("{memory_block}", mem_block).strip()
    user = f"{materials}\n\n【问题】{question}"
    payload = {
        "model": config.LLM_MODEL,
        "messages": _messages(system, user),
        "stream": False,
        "options": {
            "temperature": 0,
            "num_ctx": config.LLM_NUM_CTX,
            "num_predict": config.LLM_NUM_PREDICT,
        },
    }
    resp = _post("/api/chat", payload, timeout=timeout)
    raw = (resp.get("message") or {}).get("content") or ""
    return extract_answer(raw)


def extract_answer(raw: str) -> str:
    """取最后一个「【答案】」之后的内容；没有就用原文。"""
    idx = raw.rfind("【答案】")
    body = raw[idx + len("【答案】"):].strip() if idx >= 0 else raw.strip()
    return body or raw.strip()
