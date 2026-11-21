"""两层安全护栏。

前置 check_input：用户输入先过词表，命中 HARD 直接拒答（不检索、不调模型、不写记忆）。
后置 check_output：答案生成后再校验一次，防止模型给出法律建议 / 规避竞业 / 虚假报销指导。

词表在 sensitive_words.txt，三级：HARD（拦截）/ SOFT（放行但标记）/ ALLOW（先剔除再匹配）。
误伤策略：宁可放过——ALLOW 短语会先从问题里剔掉，保证"虚假报销的处罚是什么"这类制度查询能通过。
"""
from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from .rag.config import GUARD_ENABLED, GUARD_LLM_CHECK, SENSITIVE_WORDS_FILE

BLOCK_TEXT_INPUT = (
    "这个问题涉及违规操作或越权内容，我不能回答。"
    "制度类问题可以直接问，例如「事假怎么算」「报销谁审批」「省内出差住宿标准」。"
)
BLOCK_TEXT_OUTPUT = (
    "这个回答涉及合规风险，已被拦截。"
    "如果你想确认制度原文，可以问具体条款，例如「违规报销的处罚是什么」。"
)

# 答案侧规则：模型换个说法就漏，所以按"动作+对象"的固定搭配来匹配，不做单字判断
_OUTPUT_RULES: list[tuple[str, list[str]]] = [
    ("规避竞业", [
        "可以规避竞业", "怎么规避竞业", "规避竞业限制的方法", "不被公司发现",
        "公司查不到", "换个名字签", "挂靠别人名下",
    ]),
    ("虚假报销", [
        "建议你虚报", "可以虚报", "不开发票也能", "不用开发票也可以", "拆分开票",
        "多开点发票", "把私人消费报成", "开两张发票", "找票抵扣",
    ]),
    ("法律建议", [
        "建议你起诉", "建议你申请仲裁", "建议直接告", "可以告公司", "应赔2n",
        "属于违法解除应", "一定能赢", "仲裁必赔",
    ]),
]

_LLM_SYSTEM = (
    "你是企业知识库的合规审查员。判断助手的答案是否包含以下任一类内容：\n"
    "1. 教用户规避竞业限制或保密义务；\n"
    "2. 教用户虚假报销、虚开发票、做假账；\n"
    "3. 给出具体法律结论或诉讼建议（如「公司违法解除应赔 2N」「建议起诉」）。\n"
    "只输出一行，不要复述答案内容：\n"
    "  PASS\n"
    "  BLOCK|类别|一句话理由"
)


@dataclass
class Verdict:
    hit: bool = False
    stage: str | None = None          # "pre" / "post"
    rule: str | None = None           # "HARD|伪造|虚假报销"
    category: str | None = None
    reason: str = ""
    suspect: bool = False             # 命中 SOFT，放行但后置重点查
    mode: str = "rule"                # "rule" / "llm"


@dataclass
class _Rule:
    level: str
    category: str
    word: str
    note: str

    @property
    def norm(self) -> str:
        return _norm(self.word)

    @property
    def label(self) -> str:
        return f"{self.level}|{self.category}|{self.word}"


_table_cache: dict[str, object] = {"mtime": None, "hard": [], "soft": [], "allow": []}


# ---------------------------------------------------------------- 归一化
_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    """NFKC（全角转半角、兼容字符归一） + 去所有空白 + 小写。

    去空白是为了防"虚 假 报 销"这种拆分绕过；NFKC 顺带把全角空格、
    全角字母转掉，避免"ｉｇｎｏｒｅ"躲过英文词。
    """
    if not text:
        return ""
    s = unicodedata.normalize("NFKC", text)
    s = _WS.sub("", s)
    return s.lower()


def _mask(text: str, phrases: list[str]) -> str:
    """把白名单短语从文本里换成占位符，避免它内部的黑名单词被匹配到。"""
    for p in sorted((x for x in phrases if x), key=len, reverse=True):
        if p in text:
            text = text.replace(p, "¤")
    return text


# ---------------------------------------------------------------- 词表
def _load_table() -> tuple[list[_Rule], list[_Rule], list[_Rule]]:
    path: Path = SENSITIVE_WORDS_FILE
    try:
        mtime = path.stat().st_mtime
    except OSError:
        if _table_cache["mtime"] is not None:
            return [], [], []
        _table_cache["mtime"] = None
        return [], [], []

    if _table_cache["mtime"] == mtime:
        return _table_cache["hard"], _table_cache["soft"], _table_cache["allow"]  # type: ignore[return-value]

    hard: list[_Rule] = []
    soft: list[_Rule] = []
    allow: list[_Rule] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3:
            continue
        level, category, word = parts[0].upper(), parts[1], parts[2]
        note = parts[3] if len(parts) > 3 else ""
        if not word:
            continue
        rule = _Rule(level, category, word, note)
        {"HARD": hard, "SOFT": soft, "ALLOW": allow}.get(level, soft).append(rule)

    _table_cache.update({"mtime": mtime, "hard": hard, "soft": soft, "allow": allow})
    return hard, soft, allow


def reload_table() -> dict:
    """强制重读词表（面板改完词不想重启时用，目前只在启动和 mtime 变化时自动重读）。"""
    _table_cache["mtime"] = None
    hard, soft, allow = _load_table()
    return {"hard": len(hard), "soft": len(soft), "allow": len(allow)}


def describe() -> dict:
    hard, soft, allow = _load_table()
    return {
        "enabled": GUARD_ENABLED,
        "llm_check": GUARD_LLM_CHECK,
        "hard": len(hard),
        "soft": len(soft),
        "allow": len(allow),
        "file": str(SENSITIVE_WORDS_FILE),
    }


# ---------------------------------------------------------------- 前置
def check_input(question: str) -> Verdict:
    if not GUARD_ENABLED:
        return Verdict()

    hard, soft, allow = _load_table()
    if not (hard or soft):
        return Verdict()

    q = _norm(question)
    masked = _mask(q, [r.norm for r in allow])

    hit_hard = [r for r in hard if r.norm and r.norm in masked]
    if hit_hard:
        r = max(hit_hard, key=lambda x: len(x.norm))     # 取最长命中，定位更准
        return Verdict(hit=True, stage="pre", rule=r.label, category=r.category, reason=r.note)

    hit_soft = [r for r in soft if r.norm and r.norm in masked]
    if hit_soft:
        r = max(hit_soft, key=lambda x: len(x.norm))
        return Verdict(suspect=True, rule=r.label, category=r.category, reason=r.note)

    return Verdict()


# ---------------------------------------------------------------- 后置
def check_output(answer: str, question: str = "", suspect: bool = False) -> Verdict:
    if not GUARD_ENABLED:
        return Verdict()

    a = _norm(answer)
    for category, phrases in _OUTPUT_RULES:
        for p in phrases:
            if _norm(p) and _norm(p) in a:
                return Verdict(hit=True, stage="post", rule=f"OUT|{category}|{p}",
                               category=category, reason="答案含违规指导")

    # B 模式：只有在问题本身命中 SOFT（例如问"怎么告公司"）才值得花一次推理复核
    if suspect and GUARD_LLM_CHECK:
        return _llm_check(question, answer)

    return Verdict(suspect=suspect)


def _llm_check(question: str, answer: str) -> Verdict:
    """调一次 LLM 复核。失败一律按 PASS 处理——护栏不该把正常问答搞崩。"""
    from .rag import ollama_client

    t0 = time.time()
    try:
        raw = ollama_client.chat(
            _LLM_SYSTEM,
            f"问题：{question}\n\n助手答案：{(answer or '')[:800]}",
        )
    except Exception:  # noqa: BLE001
        return Verdict(suspect=True, mode="llm", reason="复核调用失败，按放行处理")

    text = ""
    try:
        found, text = ollama_client.split_answer(raw)
        if not found:
            text = raw
    except Exception:  # noqa: BLE001
        text = raw

    upper = (text or "").upper()
    if "BLOCK" in upper:
        parts = [p.strip() for p in (text or "").split("|")]
        category = parts[1] if len(parts) > 1 else "合规"
        reason = parts[2] if len(parts) > 2 else "LLM 复核命中"
        return Verdict(hit=True, stage="post", rule=f"LLM|{category}|复核",
                       category=category, reason=f"{reason}（{time.time() - t0:.0f}s）",
                       mode="llm")

    return Verdict(suspect=True, mode="llm", reason=f"复核通过（{time.time() - t0:.0f}s）")
