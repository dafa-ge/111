"""提示词：强约束 + 强制引用 + 无依据拒答。qwen3:4b 这类小模型，规则必须短、编号、可执行。"""
from __future__ import annotations

from .retriever import Hit

SYSTEM_PROMPT = """你是辰星科技的内部制度问答助手，回答企业员工关于公司制度的问题。

必须遵守：
1. 只能使用【参考资料】里的内容回答。不得使用常识，不得推测，不得编造条款号、金额、期限。
2. 每个结论后面必须标注引用编号，格式如 [1]；引用编号对应参考资料的编号。
3. 如果参考资料没有覆盖这个问题，直接回答"未找到相关条款"，并说明需要补充哪类文件。不要尝试作答。
4. 如果多份资料说法冲突，把两种说法都列出来，并各自标注出处。
5. 不核算个人工资、不指导社保公积金具体操作、不提供法律或税务意见。
6. 回答用中文，先给结论一句话，再分条列适用条件，总长度不超过 200 字。
7. 【用户信息】只用来判断提问背景（例如该套哪一档标准），**不得作为制度依据**，不得据记忆编造制度内容；涉及制度、金额、期限的结论必须能在【参考资料】里找到，否则拒答。
8. 思考必须极简：只用一句话判断"该查哪份资料的哪一条"，不要复述题目，不要逐字分析参考资料，不要预演答案草稿，思考不超过 80 字。
9. 最终答案的第一个字符必须是「【答案】」，紧接其后写正文，之后不要再出现任何推理或自我检查的内容。"""


def build_reference_block(hits: list[Hit]) -> str:
    lines: list[str] = []
    for i, h in enumerate(hits, start=1):
        loc = h.source + (f"，第{h.page}页" if h.page else "")
        lines.append(f"[{i}] {loc}\n{h.text}")
    return "\n\n".join(lines)


def build_memory_block(memories: list[dict]) -> str:
    lines = ["【用户信息】（来自该用户历史对话，仅作背景参考，不是制度条款）"]
    for m in memories:
        stamp = (m.get("created_at") or "")[:10]
        text = (m.get("text") or "").strip()
        if not text:
            continue
        lines.append(f"- {text}" + (f"（{stamp}）" if stamp else ""))
    return "\n".join(lines)


def build_user_message(
    question: str, hits: list[Hit], memories: list[dict] | None = None
) -> str:
    refs = build_reference_block(hits)
    parts: list[str] = []
    if memories:
        parts.append(build_memory_block(memories))
    parts.append("【参考资料】\n" + refs)
    parts.append("【问题】" + question)
    return "\n\n".join(parts)
