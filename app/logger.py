"""审计日志：每轮问答一条结构化记录。

两个 sink（都按天轮转，保留 30 天）：
    logs/audit-YYYY-MM-DD.jsonl   一行一个 JSON，给机器看、给 grep 用
    logs/app-YYYY-MM-DD.log       人类可读，直接打开看

日志只在这里写，业务代码不直接调 loguru，避免格式散落各处。
"""
from __future__ import annotations

import json
import re
import secrets
from datetime import datetime
from pathlib import Path

from loguru import logger

from .rag.config import BASE_DIR, LOG_ANSWER_MAX_CHARS, LOG_RETENTION_DAYS, LOG_TO_STDERR, LOGS_DIR

USER_INFO_JSON = BASE_DIR / "mock_data" / "user_info.json"
_DEFAULT_USER_ID = "default"

_post_cache: dict[str, object] = {"mtime": None, "map": {}}
_initialized = False


# ---------------------------------------------------------------- 初始化
def _audit_filter(record) -> bool:
    return record["extra"].get("sink") == "audit"


def _app_filter(record) -> bool:
    return record["extra"].get("sink") == "app"


def setup() -> None:
    """幂等：多次 import 只装配一次。"""
    global _initialized
    if _initialized:
        return
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(
        str(LOGS_DIR / "audit-{time:YYYY-MM-DD}.jsonl"),
        format="{message}",
        rotation="00:00",
        retention=f"{LOG_RETENTION_DAYS} days",
        encoding="utf-8",
        level="INFO",
        filter=_audit_filter,
    )
    logger.add(
        str(LOGS_DIR / "app-{time:YYYY-MM-DD}.log"),
        format="{message}",
        rotation="00:00",
        retention=f"{LOG_RETENTION_DAYS} days",
        encoding="utf-8",
        level="INFO",
        filter=_app_filter,
    )
    if LOG_TO_STDERR:
        logger.add(
            lambda m: print(m, end=""),
            format="{time:HH:mm:ss} | {level: <7} | {message}",
            level="ERROR",
        )
    _initialized = True


# ---------------------------------------------------------------- 小工具
def new_trace_id() -> str:
    """8 位十六进制，够用且好读。"""
    return secrets.token_hex(4)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _load_posts() -> dict[str, str]:
    """user_id -> 岗位。文件改了自动重读（按 mtime 判断）。"""
    try:
        mtime = USER_INFO_JSON.stat().st_mtime
    except OSError:
        return {}
    if _post_cache["mtime"] == mtime:
        return _post_cache["map"]  # type: ignore[return-value]

    mapping: dict[str, str] = {}
    try:
        data = json.loads(USER_INFO_JSON.read_text(encoding="utf-8"))
        default_id = data.get("_default_user") or ""
        for uid, info in data.items():
            if not isinstance(info, dict):
                continue
            post = str(info.get("post") or "").strip()
            if post:
                mapping[uid] = post
        if default_id in mapping:
            mapping.setdefault("default", mapping[default_id])
    except Exception:  # noqa: BLE001  档案坏了不能影响日志
        mapping = {}
    _post_cache["mtime"] = mtime
    _post_cache["map"] = mapping
    return mapping


def resolve_post(user_id: str) -> str:
    return _load_posts().get(user_id, "") or "-"


_PHONE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
_IDCARD = re.compile(r"(?<!\w)\d{17}[\dXx](?!\w)")


def _scrub(text: str) -> str:
    """手机号与身份证号不进日志。"""
    text = _PHONE.sub("1**********", text)
    return _IDCARD.sub("******************", text)


def _clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + f"…（共{len(text)}字）"


def _hits_to_log(hits) -> list[dict]:
    """只记文件名与分数，不记原文——否则日志会被切片正文撑爆。"""
    out: list[dict] = []
    for h in hits or []:
        if isinstance(h, dict):
            out.append({
                "source": h.get("source"),
                "cos": h.get("cos"),
                "bm25": h.get("bm25"),
            })
        else:
            out.append({
                "source": getattr(h, "source", None),
                "cos": round(float(getattr(h, "cos_sim", 0)), 3),
                "bm25": round(float(getattr(h, "bm25", 0)), 2),
            })
    return out


def _tools_to_log(tool_calls) -> list[dict]:
    """兼容两种形状：{"name","args","ok"} 与 agent 的 {"name","args","result":{...}}。"""
    out: list[dict] = []
    for t in tool_calls or []:
        if not isinstance(t, dict):
            out.append({"name": str(t)})
            continue
        inner = t.get("result")
        ok = bool(inner.get("ok", True)) if isinstance(inner, dict) else bool(t.get("ok", True))
        out.append({
            "name": t.get("name"),
            "args": t.get("args") or {},
            "ok": ok,
        })
    return out


def _memories_to_log(memories) -> list[str]:
    return [m.get("text", "") if isinstance(m, dict) else str(m) for m in (memories or [])]


# ---------------------------------------------------------------- 主入口
def log_turn(
    *,
    question: str,
    answer: str,
    user_id: str = _DEFAULT_USER_ID,
    mode: str = "rag",
    session_id: str = "",
    route: str | None = None,
    hits=None,
    tool_calls=None,
    memories=None,
    blocked: bool = False,
    stage: str | None = None,
    hit_rule: str | None = None,
    suspect: bool = False,
    answer_raw: str | None = None,
    elapsed: float = 0.0,
    trace_id: str | None = None,
    error: str | None = None,
) -> str:
    """写一轮问答的审计记录，返回 trace_id（面板会带给前端）。"""
    setup()
    trace_id = trace_id or new_trace_id()
    post = resolve_post(user_id)

    record = {
        "ts": _now(),
        "trace_id": trace_id,
        "user_id": user_id,
        "post": post,
        "mode": mode,
        "session_id": session_id,
        "question": _scrub(_clip(question, 200)),
        "route": route,
        "hits": _hits_to_log(hits),
        "tool_calls": _tools_to_log(tool_calls),
        "memories": _memories_to_log(memories),
        "blocked": bool(blocked),
        "stage": stage,
        "hit_rule": hit_rule,
        "suspect": bool(suspect),
        "answer": _scrub(_clip(answer, LOG_ANSWER_MAX_CHARS)),
        "answer_raw": _scrub(_clip(answer_raw, LOG_ANSWER_MAX_CHARS)) if answer_raw else None,
        "elapsed": round(float(elapsed or 0), 1),
        "error": error,
    }

    logger.bind(sink="audit").info(json.dumps(record, ensure_ascii=False))

    flag = "拦截" if blocked else ("可疑" if suspect else "通过")
    route_s = f" route={route}" if route else ""
    logger.bind(sink="app").info(
        f"{record['ts']} | {flag} | trace={trace_id} user={user_id}/{post} "
        f"mode={mode}{route_s} elapsed={record['elapsed']}s "
        f"hits={len(record['hits'])} tools={len(record['tool_calls'])}"
        + (f" stage={stage} rule={hit_rule}" if blocked else "")
        + f"\n    Q: {record['question']}\n    A: {record['answer']}"
    )
    return trace_id


def log_event(kind: str, **fields) -> None:
    """非问答事件（登录、权限拒绝）只写 app.log，不污染 audit jsonl。"""
    setup()
    detail = " ".join(f"{k}={v}" for k, v in fields.items() if v not in (None, ""))
    logger.bind(sink="app").info(f"{_now()} | {kind} | {detail}")


def recent_records(limit: int = 10) -> list[dict]:
    """面板审计卡片用：读最新的 audit jsonl，按时间倒序取最近 N 条（拦截与通过都返回）。"""
    setup()
    items: list[dict] = []
    files = sorted(LOGS_DIR.glob("audit-*.jsonl"), reverse=True)
    for path in files:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in reversed(lines):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            items.append(rec)
            if len(items) >= limit:
                return items
    return items
