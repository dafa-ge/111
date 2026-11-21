"""从用户话里抽取稳定事实。

规则优先（毫秒级）。LLM 兜底默认关闭：qwen3:4b 一次总结要 6~10 分钟，
还会和前台问答抢 Ollama，规则覆盖 demo 需要的五类事实就够。

抽出来的每条事实都要能在用户原话里找到字面依据，找不到的直接丢，
避免把模型编的内容写进长期记忆。
"""
from __future__ import annotations

import re
import time
import unicodedata

ROLE_WORDS = "市场|销售|商务|技术|研发|开发|测试|运维|职能|行政|人事|财务|产品|运营"
CITIES = ("北京|上海|广州|深圳|杭州|成都|武汉|南京|西安|重庆|苏州|"
          "长沙|天津|郑州|青岛|厦门|合肥|济南|东莞|佛山|宁波|无锡")

RE_ROLE = re.compile(
    rf"(?:我是|我属于|我在|我这边是)\s*(?:一名|一个)?\s*({ROLE_WORDS})\s*(岗位|岗|部|部门)"
)
RE_DATE = re.compile(r"(20\d{2})\s*[年\-/\.]\s*(\d{1,2})?\s*月?\s*(?:份|初|底)?\s*入职")
RE_TENURE = re.compile(r"入职\s*(?:已经|有)?\s*(\d+)\s*年")
RE_DEPT = re.compile(r"我在\s*([\u4e00-\u9fa5]{2,10}?)\s*部门")
RE_CITY = re.compile(rf"(?:常驻|base\s*在|工作在|我在)\s*(?:{CITIES})(?:市)?", re.I)
RE_CITY_VALUE = re.compile(rf"({CITIES})")
RE_NAME = re.compile(r"(?:我叫|我的名字是|称呼我|可以叫我)\s*([\u4e00-\u9fa5]{2,4})")

MAX_FACTS = 4


def normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "").strip()


def _literal_ok(value: str, source: str) -> bool:
    """事实里的关键值必须原样出现在用户输入里。"""
    return bool(value) and value in source


def extract(text: str) -> list[dict]:
    """返回 [{"kind": "岗位", "text": "岗位：市场岗"}, ...]，最多 4 条。"""
    src = normalize(text)
    if not src:
        return []

    facts: list[dict] = []
    seen: set[str] = set()

    def add(kind: str, value: str, text_out: str) -> None:
        if len(facts) >= MAX_FACTS:
            return
        key = f"{kind}|{text_out}"
        if key in seen or not _literal_ok(value, src):
            return
        seen.add(key)
        facts.append({"kind": kind, "text": text_out})

    for m in RE_ROLE.finditer(src):
        role, tail = m.group(1), m.group(2)
        if tail in ("部", "部门"):
            add("部门", role, f"部门：{role}部")
        else:
            add("岗位", role, f"岗位：{role}岗")

    dm = RE_DATE.search(src)
    if dm:
        year, month = dm.group(1), dm.group(2)
        value = f"{year}"
        out = f"入职时间：{year}年" + (f"{int(month)}月" if month else "")
        add("入职时间", value, out)

    tm = RE_TENURE.search(src)
    if tm and not dm:
        n = int(tm.group(1))
        if 0 < n < 40:
            year = time.localtime().tm_year - n
            add("入职时间", str(year), f"入职时间：约{year}年（入职满{n}年）")

    dept = RE_DEPT.search(src)
    if dept:
        add("部门", dept.group(1), f"部门：{dept.group(1)}部")

    city = RE_CITY.search(src)
    if city:
        cm = RE_CITY_VALUE.search(city.group(0))
        if cm:
            add("常驻城市", cm.group(1), f"常驻城市：{cm.group(1)}")

    name = RE_NAME.search(src)
    if name:
        add("称呼", name.group(1), f"称呼：{name.group(1)}")

    return facts


def should_llm_fallback(text: str) -> bool:
    """规则完全没命中、且这句话是在讲自己，才值得再花一次模型调用。"""
    src = normalize(text)
    if not src or extract(src):
        return False
    return "我" in src and len(src) <= 80


def summarize_with_llm(text: str) -> list[dict]:
    """LLM 兜底。默认不启用（见 config.MEMORY_LLM_FALLBACK）。"""
    from ..rag import ollama_client

    system = (
        "你只做事实抽取，不做问答。只抽取用户明确说出的稳定个人信息：岗位、部门、"
        "入职时间、常驻城市、称呼。输出格式：第一行【答案】；其后每行一条，形如"
        "“岗位：市场岗”；没有可记事实就只输出【答案】无。不要推测，不要复述别的。"
    )
    try:
        raw = ollama_client.chat(system, f"用户说：{text}", timeout=900)
    except Exception:  # noqa: BLE001
        return []

    facts: list[dict] = []
    src = normalize(text)
    for line in raw.splitlines():
        line = line.strip(" -·\t")
        if not line or line == "无" or "：" not in line:
            continue
        kind, value = line.split("：", 1)
        kind, value = kind.strip(), value.strip()
        if kind not in {"岗位", "部门", "入职时间", "常驻城市", "称呼"}:
            continue
        if not _literal_ok(value.rstrip("岗部"), src):
            continue
        facts.append({"kind": kind, "text": f"{kind}：{value}"})
        if len(facts) >= MAX_FACTS:
            break
    return facts
