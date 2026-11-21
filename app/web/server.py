"""本地操作面板：状态、提问（含流式）、资料上传与入库、报告查看。只绑 127.0.0.1。"""
from __future__ import annotations

import json
import os
import queue
import re
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from ..memory.extractor import extract
from ..memory.long_memory import get_long_memory
from ..memory.short_memory import ShortMemory, get_short_memory
from ..rag import ingest, ollama_client, prompt, service, vector_store
from ..rag.config import (
    AUTH_ENABLED, CHUNKS_JSON, DOCS_DIR, EMBED_MODEL, LLM_MODEL, MANIFEST_JSON,
    SUPPORTED_EXTS, TOP_N_FINAL,
)
from ..rag.qa import NO_ANSWER
from ..rag.retriever import is_confident
from ..agent.graph import GRAPH as AGENT_GRAPH
from ..auth import (
    AuthMiddleware, CurrentUser, load_users, login as auth_login, require_admin, require_user,
)
from ..guardrail import (
    BLOCK_TEXT_INPUT, BLOCK_TEXT_OUTPUT, check_input, check_output,
    describe as guard_describe,
)
from ..logger import log_event, log_turn, new_trace_id, recent_records
from .jobs import JobRunner

USER_ID = "default"          # 仅鉴权关闭（RAG_AUTH_ENABLED=0）时的兜底身份
DEFAULT_SESSION = "panel-default"

app = FastAPI(title="RAG 知识库面板", docs_url=None, redoc_url=None)
app.add_middleware(AuthMiddleware)      # 解析 token → request.state.user；白名单放行
runner = JobRunner()
STATIC = Path(__file__).parent / "static"

MAX_UPLOAD = 20 * 1024 * 1024
NAME_RE = re.compile(
    r"^(?P<topic>.+?)_(?:制度|流程|SOP|方案|合同|纪要|报告|台账|模板|证照|手册|复盘)_"
    r"(?P<date>\d{8})_v(?P<ver>\d+\.\d+)$"
)
# qwen3 思考结束标记（全角竖线 U+FF5C、分隔符 U+2581）
_THINK_ENDS = (
    "<" + "\uff5c" + "end" + "\u2581" + "of" + "\u2581" + "thinking" + "\uff5c" + ">",
    "</think>",
    "</thinking>",
)
_char_cache: dict[tuple[str, float], int] = {}


# ---------------------------------------------------------------- 工具
def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _hit_json(h) -> dict:
    return {
        "source": h.source,
        "page": h.page,
        "section": h.section,
        "cos": round(h.cos_sim, 3),
        "bm25": round(h.bm25, 2),
        "neighbor": h.neighbor,
        "text": h.text[:600],
    }


def _session_of(payload: dict, user: CurrentUser | None = None) -> str:
    """会话号加用户前缀，避免两个员工用同一个 session_id 串上下文。

    前缀用 `-`（SESSION_ID_RE 只允许字母数字下划线连字符）。
    """
    sid = (payload.get("session_id") or DEFAULT_SESSION).strip()
    if not ShortMemory.valid_id(sid):
        raise HTTPException(status_code=400, detail="非法 session_id")
    if user and user.id != "default":
        prefix = f"{user.id}-"
        if not sid.startswith(prefix):
            sid = prefix + sid
            if not ShortMemory.valid_id(sid):
                raise HTTPException(status_code=400, detail="会话号过长")
    return sid


def _remember_turn(session_id: str, question: str, answer: str,
                   facts: list[dict] | None = None, *, allow_extract: bool = True,
                   user_id: str = USER_ID) -> list[dict]:
    """一轮结束：写短期会话 + 抽稳定事实写长期记忆。异常一律不影响问答。

    allow_extract=False 用于被拦截的轮次——安全文案进短期上下文可以，
    但不能因此把违规问句抽出事实写进长期记忆。
    """
    if facts is None:
        facts = []
    try:
        short = get_short_memory()
        short.append(session_id, "user", question)
        short.append(session_id, "assistant", answer)
        if allow_extract and not facts:
            facts = extract(question)
        if facts:
            get_long_memory().remember(user_id, session_id, facts)
    except Exception:  # noqa: BLE001
        pass
    return facts


def _chars_of(path: Path, size: int) -> int:
    key = (path.name, path.stat().st_mtime)
    if key in _char_cache:
        return _char_cache[key]
    chars = 0
    if path.suffix.lower() in {".txt", ".md"}:
        for enc in ("utf-8-sig", "utf-8", "gb18030"):
            try:
                chars = len(path.read_text(encoding=enc))
                break
            except UnicodeDecodeError:
                continue
    else:
        try:
            from ..rag.loader import load_file

            chars = load_file(path).char_count
        except Exception:  # noqa: BLE001
            chars = -1
    _char_cache[key] = chars
    return chars


def _suggest_name(path: Path) -> str:
    stem = re.sub(r"[^\w\u4e00-\u9fff-]", "", path.stem)[:20] or "新文档"
    return f"{stem}_制度_{datetime.now():%Y%m%d}_v1.0{path.suffix}"


def _memory_stats(user_id: str = USER_ID) -> dict:
    try:
        mem = get_long_memory()
        return {"count": mem.count(user_id), "ttl_days": mem.ttl_days}
    except Exception:  # noqa: BLE001
        return {"count": 0, "ttl_days": 0}


def list_files() -> list[dict]:
    manifest = ingest.load_manifest()
    items: list[dict] = []
    if not DOCS_DIR.exists():
        return items
    for p in sorted(DOCS_DIR.iterdir()):
        if not p.is_file() or p.suffix.lower() not in SUPPORTED_EXTS:
            continue
        st = p.stat()
        chars = _chars_of(p, st.st_size)
        digest = ingest.file_sha256(p) if st.st_size else ""
        prev = manifest.get(p.name)
        if st.st_size == 0 or chars == 0:
            status = "empty"
        elif prev and prev.get("sha256") == digest:
            status = "indexed"
        elif prev:
            status = "changed"
        else:
            status = "new"
        naming_ok = bool(NAME_RE.match(p.stem))
        items.append(
            {
                "name": p.name,
                "size": st.st_size,
                "chars": chars,
                "status": status,
                "chunks": (prev or {}).get("chunks"),
                "ingested_at": (prev or {}).get("ingested_at"),
                "naming_ok": naming_ok,
                "suggest": None if naming_ok else _suggest_name(p),
                "modified": datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M"),
            }
        )
    return items


# ---------------------------------------------------------------- 接口
@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


# ---------------------------------------------------------------- 鉴权
@app.post("/api/auth/login")
def auth_login_endpoint(payload: dict):
    account = (payload.get("account") or "").strip()
    password = payload.get("password") or ""
    if not account or not password:
        raise HTTPException(status_code=400, detail="账号和密码都不能为空")
    res = auth_login(account, password)
    if not res:
        # 不区分"账号不存在"和"密码错"，避免被枚举
        log_event("登录失败", account=account)
        raise HTTPException(status_code=401, detail="账号或密码不正确")
    log_event("登录成功", account=account, role=res["user"]["role"])
    return res


@app.get("/api/auth/me")
def auth_me(user: CurrentUser = Depends(require_user)):
    return {"user": user.as_dict(), "auth_enabled": AUTH_ENABLED}


@app.post("/api/auth/logout")
def auth_logout(user: CurrentUser = Depends(require_user)):
    """JWT 无状态，服务端不维护会话；前端清 token 即可，这里只记一笔审计。"""
    log_event("登出", account=user.id, role=user.role)
    return {"ok": True}


@app.get("/api/status")
def status(user: CurrentUser = Depends(require_user)):
    health = ollama_client.health_check()
    backend, reason = vector_store.decide_backend()
    try:
        chunks = len(ingest.load_chunks())
        vectors = vector_store.get_store().count()
    except Exception:  # noqa: BLE001
        chunks, vectors = 0, 0

    files = list_files()
    payload = {
        "ollama": health,
        "backend": backend,
        "backend_reason": reason,
        "embed_model": EMBED_MODEL,
        "llm_model": LLM_MODEL,
        "docs_dir": str(DOCS_DIR),
        "chunks": chunks,
        "vectors": vectors,
        "consistent": chunks == vectors,
        "files": files,
        "job": runner.snapshot(),
        "busy": runner.is_busy(),
        "memory": _memory_stats(user.id),
        "guard": guard_describe(),
        "user": user.as_dict(),
        "auth_enabled": AUTH_ENABLED,
    }

    if not user.is_admin:
        # 员工看不到知识库明细（42 个文件名不返回），只给数量
        ingested = sum(1 for f in files if f["status"] == "indexed")
        payload.pop("files", None)
        payload.pop("docs_dir", None)
        payload["files_count"] = len(files)
        payload["files_ingested"] = ingested
        payload["pending"] = len(files) - ingested
        g = payload.get("guard") or {}
        payload["guard"] = {
            "enabled": g.get("enabled"), "hard": g.get("hard"),
            "soft": g.get("soft"), "allow": g.get("allow"),
        }
    return payload


@app.get("/api/guard")
def guard_info(user: CurrentUser = Depends(require_admin)):
    """面板「审计记录」卡片用。员工看不到拦截明细。"""
    return {"stats": guard_describe(), "recent": recent_records(10)}


@app.post("/api/ask")
def ask(payload: dict, user: CurrentUser = Depends(require_user)):
    question = (payload.get("question") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")
    if runner.is_busy():
        raise HTTPException(status_code=409, detail="入库任务进行中，请稍后再问")

    session_id = _session_of(payload, user)
    started = time.time()
    trace_id = new_trace_id()

    guard = check_input(question)
    if guard.hit:
        # 前置拦截：不检索、不调模型、不写记忆，只落一条审计
        log_turn(question=question, answer=BLOCK_TEXT_INPUT, user_id=user.id, mode="rag",
                 session_id=session_id, blocked=True, stage="pre", hit_rule=guard.rule,
                 trace_id=trace_id, elapsed=time.time() - started)
        return {"answer": BLOCK_TEXT_INPUT, "confident": False, "blocked": True,
                "block_reason": guard.rule, "hits": [], "memories": [], "facts": [],
                "trace_id": trace_id, "elapsed": round(time.time() - started, 1)}

    short = get_short_memory()
    long_mem = get_long_memory()
    try:
        memories = long_mem.recall(user.id, question)
        history = short.messages_for_llm(session_id)[-4:] or None
    except Exception:  # noqa: BLE001
        memories, history = [], None

    retriever = service.get_retriever()
    hits = retriever.search(question, top_n=int(payload.get("top_n") or TOP_N_FINAL))
    confident = is_confident(hits)
    answer = NO_ANSWER
    if confident:
        try:
            answer = ollama_client.chat(
                prompt.SYSTEM_PROMPT,
                prompt.build_user_message(question, hits, memories),
                history=history,
            )
        except ollama_client.OllamaError as e:
            log_turn(question=question, answer="", user_id=user.id, mode="rag",
                     session_id=session_id, hits=hits, memories=memories,
                     trace_id=trace_id, elapsed=time.time() - started, error=str(e))
            return {
                "answer": "",
                "confident": confident,
                "error": str(e),
                "hits": [_hit_json(h) for h in hits],
                "memories": memories,
                "trace_id": trace_id,
                "elapsed": round(time.time() - started, 1),
            }

    facts = extract(question) if not confident else []
    if facts:
        # 自述类输入：回"已记住"，而不是"未找到条款"
        answer = ("已记住：" + "、".join(f["text"] for f in facts)
                  + "。之后提问我会按这个身份回答。")

    guard_out = check_output(answer, question, suspect=guard.suspect)
    answer_raw = None
    if guard_out.hit:
        answer_raw, answer = answer, BLOCK_TEXT_OUTPUT
        facts = []  # 被拦的轮次不抽事实写长期记忆
    facts = _remember_turn(session_id, question, answer, facts,
                           allow_extract=not guard_out.hit, user_id=user.id)
    log_turn(question=question, answer=answer, user_id=user.id, mode="rag",
             session_id=session_id, route="direct" if confident else "reject",
             hits=hits, memories=memories, blocked=guard_out.hit,
             stage=guard_out.stage, hit_rule=guard_out.rule,
             suspect=guard.suspect or guard_out.suspect, answer_raw=answer_raw,
             trace_id=trace_id, elapsed=time.time() - started)
    return {
        "answer": answer,
        "confident": confident,
        "hits": [_hit_json(h) for h in hits],
        "memories": memories,
        "facts": facts,
        "blocked": guard_out.hit,
        "block_reason": guard_out.rule,
        "trace_id": trace_id,
        "elapsed": round(time.time() - started, 1),
    }


@app.post("/api/ask/stream")
def ask_stream(payload: dict, user: CurrentUser = Depends(require_user)):
    question = (payload.get("question") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")
    session_id = _session_of(payload, user)

    def gen():
        started = time.time()
        trace_id = new_trace_id()

        guard = check_input(question)
        if guard.hit:
            yield _sse("sources", {"hits": [], "confident": False, "memories": [],
                                   "blocked": True})
            yield _sse("answer", {"delta": BLOCK_TEXT_INPUT})
            log_turn(question=question, answer=BLOCK_TEXT_INPUT, user_id=user.id, mode="rag",
                     session_id=session_id, blocked=True, stage="pre",
                     hit_rule=guard.rule, trace_id=trace_id, elapsed=time.time() - started)
            yield _sse("done", {"elapsed": round(time.time() - started, 1),
                                "confident": False, "blocked": True,
                                "block_reason": guard.rule, "trace_id": trace_id})
            return

        short = get_short_memory()
        long_mem = get_long_memory()
        try:
            memories = long_mem.recall(user.id, question)
            history = short.messages_for_llm(session_id)[-4:] or None
        except Exception:  # noqa: BLE001  记忆故障不能挡住问答
            memories, history = [], None

        try:
            retriever = service.get_retriever()
            hits = retriever.search(question)
        except Exception as e:  # noqa: BLE001
            log_turn(question=question, answer="", user_id=user.id, mode="rag",
                     session_id=session_id, trace_id=trace_id,
                     elapsed=time.time() - started, error=str(e))
            yield _sse("error", {"message": str(e)})
            return

        confident = is_confident(hits)
        yield _sse("sources", {
            "hits": [_hit_json(h) for h in hits],
            "confident": confident,
            "memories": memories,
        })

        if not confident:
            facts = extract(question)
            if facts:
                # 自述类输入（"我是市场岗"）：不是在问制度，别回"未找到条款"
                msg = ("已记住：" + "、".join(f["text"] for f in facts)
                       + "。之后提问我会按这个身份回答。")
            else:
                msg = NO_ANSWER
            guard_out = check_output(msg, question, suspect=guard.suspect)
            answer_raw = None
            if guard_out.hit:
                answer_raw, msg = msg, BLOCK_TEXT_OUTPUT
                facts = []
            _remember_turn(session_id, question, msg, facts,
                           allow_extract=not guard_out.hit, user_id=user.id)
            log_turn(question=question, answer=msg, user_id=user.id, mode="rag",
                     session_id=session_id, route="reject", hits=hits, memories=memories,
                     blocked=guard_out.hit, stage=guard_out.stage, hit_rule=guard_out.rule,
                     suspect=guard.suspect or guard_out.suspect, answer_raw=answer_raw,
                     trace_id=trace_id, elapsed=time.time() - started)
            yield _sse("answer", {"delta": msg})
            yield _sse("done", {"elapsed": round(time.time() - started, 1),
                                "confident": False, "facts": facts,
                                "blocked": guard_out.hit, "block_reason": guard_out.rule,
                                "trace_id": trace_id})
            return

        user_msg = prompt.build_user_message(question, hits, memories)
        buffer = ""
        last_beat = 0.0
        # 不逐块切分：qwen3 会在思考里演练标记，流式阶段无法判断哪一次才是真结束。
        # 所以流式只负责"思考心跳"，答案在流结束时一次性取出。
        try:
            for _kind, text in ollama_client.chat_stream(
                prompt.SYSTEM_PROMPT, user_msg, history=history
            ):
                buffer += text
                now = time.time()
                if now - last_beat >= 2:
                    last_beat = now
                    yield _sse("thinking", {"seconds": int(now - started)})
        except Exception as e:  # noqa: BLE001
            yield _sse("error", {"message": f"{type(e).__name__}: {e}"})
            return

        found, answer = ollama_client.split_answer(buffer)
        if not (found and answer.strip()):
            log_turn(question=question, answer="", user_id=user.id, mode="rag",
                     session_id=session_id, hits=hits, memories=memories,
                     trace_id=trace_id, elapsed=time.time() - started,
                     error="思考把上下文用满了，没写出最终答案")
            yield _sse("error", {
                "message": "模型思考把上下文用满了，没写出最终答案。已把上限调到 8K/4096，请重试；"
                           "仍失败说明这模型思考太长，建议换 qwen2.5:3b。"})
            yield _sse("done", {"elapsed": round(time.time() - started, 1),
                                "confident": False, "failed": True, "trace_id": trace_id})
            return

        answer = answer.strip()
        guard_out = check_output(answer, question, suspect=guard.suspect)
        answer_raw = None
        if guard_out.hit:
            answer_raw, answer = answer, BLOCK_TEXT_OUTPUT
        yield _sse("answer", {"delta": answer})
        facts = _remember_turn(session_id, question, answer,
                               allow_extract=not guard_out.hit, user_id=user.id)
        log_turn(question=question, answer=answer, user_id=user.id, mode="rag",
                 session_id=session_id, route="direct", hits=hits, memories=memories,
                 blocked=guard_out.hit, stage=guard_out.stage, hit_rule=guard_out.rule,
                 suspect=guard.suspect or guard_out.suspect, answer_raw=answer_raw,
                 trace_id=trace_id, elapsed=time.time() - started)
        yield _sse("done", {"elapsed": round(time.time() - started, 1),
                            "confident": True, "facts": facts,
                            "blocked": guard_out.hit, "block_reason": guard_out.rule,
                            "trace_id": trace_id})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/agent/ask/stream")
def agent_ask_stream(payload: dict, user: CurrentUser = Depends(require_user)):
    """智能体模式：AI 自己判断走 直接回答 / 查文档 / 调工具。

    事件流：sources → step（每完成一个节点推一条）→ answer → done
    图在后台线程跑，主线程每 2 秒推心跳，避免 compose 阶段 5~10 分钟静默。
    """
    question = (payload.get("question") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")
    session_id = _session_of(payload, user)

    def gen():
        started = time.time()
        trace_id = new_trace_id()

        guard = check_input(question)
        if guard.hit:
            yield _sse("sources", {"hits": [], "confident": False, "memories": [],
                                   "agent": True, "blocked": True})
            yield _sse("answer", {"delta": BLOCK_TEXT_INPUT})
            log_turn(question=question, answer=BLOCK_TEXT_INPUT, user_id=user.id, mode="agent",
                     session_id=session_id, blocked=True, stage="pre",
                     hit_rule=guard.rule, trace_id=trace_id, elapsed=time.time() - started)
            yield _sse("done", {"elapsed": round(time.time() - started, 1),
                                "confident": False, "blocked": True,
                                "block_reason": guard.rule, "trace_id": trace_id,
                                "agent": True})
            return

        short = get_short_memory()
        long_mem = get_long_memory()
        try:
            memories = long_mem.recall(user.id, question)
            history = short.messages_for_llm(session_id)[-4:] or None
        except Exception:  # noqa: BLE001
            memories, history = [], None

        init: dict = {"question": question, "session_id": session_id, "user_id": user.id,
                      "memories": memories, "history": history, "steps": []}
        final: dict = dict(init)
        box: queue.Queue = queue.Queue()

        def worker() -> None:
            try:
                for chunk in AGENT_GRAPH.stream(init, stream_mode="updates",
                                                config={"recursion_limit": 8}):
                    for _node, upd in chunk.items():
                        if isinstance(upd, dict):
                            box.put(("update", upd))
                box.put(("end", None))
            except Exception as e:  # noqa: BLE001
                box.put(("error", f"{type(e).__name__}: {e}"))

        threading.Thread(target=worker, daemon=True).start()
        yield _sse("sources", {"hits": [], "confident": True, "memories": memories,
                               "agent": True})

        while True:
            try:
                kind, data = box.get(timeout=2)
            except queue.Empty:
                yield _sse("thinking", {"seconds": int(time.time() - started)})
                continue
            if kind == "error":
                log_turn(question=question, answer="", user_id=user.id, mode="agent",
                         session_id=session_id, memories=memories, trace_id=trace_id,
                         elapsed=time.time() - started, error=str(data))
                yield _sse("error", {"message": data})
                yield _sse("done", {"elapsed": round(time.time() - started, 1),
                                    "confident": False, "failed": True,
                                    "trace_id": trace_id, "agent": True})
                return
            if kind == "end":
                break
            for s in data.get("steps", []):
                yield _sse("step", s)
            final.update({k: v for k, v in data.items() if k != "steps"})
            if data.get("answer"):
                yield _sse("answer", {"delta": data["answer"]})

        answer = final.get("answer") or NO_ANSWER
        guard_out = check_output(answer, question, suspect=guard.suspect)
        answer_raw = None
        if guard_out.hit:
            # 注意：answer 事件已经推出去了，这里补一条 blocked 标记让前端替换展示
            answer_raw, answer = answer, BLOCK_TEXT_OUTPUT
            yield _sse("blocked", {"delta": BLOCK_TEXT_OUTPUT, "reason": guard_out.rule})
        facts = _remember_turn(session_id, question, answer,
                               allow_extract=not guard_out.hit, user_id=user.id)
        log_turn(question=question, answer=answer, user_id=user.id, mode="agent",
                 session_id=session_id, route=final.get("route"),
                 hits=final.get("rag_hits"), tool_calls=final.get("tool_results"),
                 memories=memories, blocked=guard_out.hit,
                 stage=guard_out.stage, hit_rule=guard_out.rule,
                 suspect=guard.suspect or guard_out.suspect, answer_raw=answer_raw,
                 trace_id=trace_id, elapsed=time.time() - started)
        yield _sse("done", {"elapsed": round(time.time() - started, 1),
                            "confident": bool(final.get("answer")),
                            "facts": facts, "route": final.get("route"), "agent": True,
                            "blocked": guard_out.hit, "block_reason": guard_out.rule,
                            "trace_id": trace_id})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/docs/upload")
async def upload(file: UploadFile = File(...), overwrite: bool = Form(False),
                 user: CurrentUser = Depends(require_admin)):
    original = file.filename or ""
    if not original:
        raise HTTPException(status_code=400, detail="文件名为空")
    if any(ch in original for ch in ("/", "\\")) or ".." in original:
        raise HTTPException(status_code=400, detail="文件名不能包含路径分隔符或 ..")
    raw_name = Path(original).name
    ext = Path(raw_name).suffix.lower()
    if ext not in SUPPORTED_EXTS:
        raise HTTPException(status_code=415, detail=f"不支持 {ext}，仅支持 txt / md / pdf / docx")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="文件为空（0 字节），未入库")
    if len(content) > MAX_UPLOAD:
        raise HTTPException(status_code=413, detail="超过 20 MB 上限")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    target = (DOCS_DIR / raw_name).resolve()
    if target.parent != DOCS_DIR.resolve():
        raise HTTPException(status_code=400, detail="非法的文件路径")

    if target.exists() and not overwrite:
        return JSONResponse(
            status_code=409,
            content={"exists": True, "message": "同名文件已存在", "name": raw_name},
        )

    target.write_bytes(content)
    queued = runner.submit(f"入库 {raw_name}", lambda: ingest.ingest())
    return {"ok": True, "name": raw_name, "size": len(content), "queued": queued}


@app.post("/api/docs/reingest")
def reingest(payload: dict, user: CurrentUser = Depends(require_admin)):
    source = Path(payload.get("source") or "").name
    if not (DOCS_DIR / source).exists():
        raise HTTPException(status_code=404, detail="文件不存在")

    def job():
        manifest = ingest.load_manifest()
        manifest.pop(source, None)
        ingest.save_json(MANIFEST_JSON, manifest)
        try:
            vector_store.get_store().delete(where={"source": source})
        except Exception:  # noqa: BLE001
            pass
        return ingest.ingest()

    if not runner.submit(f"重入库 {source}", job):
        raise HTTPException(status_code=409, detail="有任务正在运行，请稍后")
    return {"ok": True, "name": source}


@app.post("/api/docs/remove")
def remove_from_index(payload: dict, user: CurrentUser = Depends(require_admin)):
    """只从索引移除，不删源文件。走任务队列，避免与入库并发写 manifest。"""
    source = Path(payload.get("source") or "").name
    if not source:
        raise HTTPException(status_code=400, detail="缺少 source")

    def job():
        manifest = ingest.load_manifest()
        manifest.pop(source, None)
        ingest.save_json(MANIFEST_JSON, manifest)
        try:
            vector_store.get_store().delete(where={"source": source})
        except Exception:  # noqa: BLE001
            pass
        chunks = [c for c in ingest.load_chunks() if c.get("source") != source]
        ingest.save_json(CHUNKS_JSON, chunks)
        return {"removed": source, "chunks_in_file": len(chunks)}

    if not runner.submit(f"移出索引 {source}", job):
        raise HTTPException(status_code=409, detail="有任务正在运行，请稍后")
    return {"ok": True, "queued": True, "note": "已排队从索引移除，源文件保留"}


@app.post("/api/build")
def build(payload: dict | None = None, user: CurrentUser = Depends(require_admin)):
    rebuild = bool((payload or {}).get("rebuild"))
    label = "全量重建" if rebuild else "增量入库"
    if not runner.submit(label, lambda: ingest.ingest(rebuild=rebuild)):
        raise HTTPException(status_code=409, detail="有任务正在运行，请稍后")
    return {"ok": True, "label": label}


@app.get("/api/job")
def job(user: CurrentUser = Depends(require_user)):
    return runner.snapshot()


@app.post("/api/session/new")
def session_new(payload: dict | None = None, user: CurrentUser = Depends(require_user)):
    """开新会话：清短期上下文，长期记忆不动（这样才能演示"跨会话记得你"）。"""
    sid = (payload or {}).get("session_id") or DEFAULT_SESSION
    try:
        sid = _session_of({"session_id": sid}, user)
    except HTTPException:
        sid = f"{user.id}-{DEFAULT_SESSION}" if user.id != "default" else DEFAULT_SESSION
    get_short_memory().clear(sid)
    return {"ok": True, "session_id": sid}


@app.get("/api/memory/list")
def memory_list(user_id: str | None = None, user: CurrentUser = Depends(require_user)):
    target = (user_id or "").strip() or user.id
    if target != user.id and not user.is_admin:
        raise HTTPException(status_code=403, detail="只能查看自己的记忆")
    mem = get_long_memory()
    return {"count": mem.count(target), "ttl_days": mem.ttl_days,
            "items": mem.list_all(target), "user_id": target}


@app.post("/api/memory/delete")
def memory_delete(payload: dict, user: CurrentUser = Depends(require_user)):
    mid = (payload.get("id") or "").strip()
    if not mid:
        raise HTTPException(status_code=400, detail="缺少 id")
    mem = get_long_memory()
    if not user.is_admin:
        # 员工只能删自己的；不区分"不存在"和"不属于你"，避免被用来探测别人的记忆 id
        owned = {m["id"] for m in mem.list_all(user.id)}
        if mid not in owned:
            raise HTTPException(status_code=403, detail="只能删除自己的记忆")
    return {"ok": mem.delete(mid)}


@app.post("/api/memory/purge")
def memory_purge(payload: dict | None = None, user: CurrentUser = Depends(require_user)):
    days = (payload or {}).get("days")
    # 员工只清自己的，管理员清全库
    target = None if user.is_admin else user.id
    return {"ok": True, "removed": get_long_memory().purge_expired(days, user_id=target)}


@app.get("/api/reports")
def reports(user: CurrentUser = Depends(require_admin)):
    from ..rag.config import REPORTS_DIR

    if not REPORTS_DIR.exists():
        return {"items": []}
    items = []
    for p in sorted(REPORTS_DIR.glob("*.md"), key=lambda x: x.stat().st_mtime, reverse=True)[:6]:
        items.append(
            {
                "name": p.name,
                "mtime": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                "text": p.read_text(encoding="utf-8")[:4000],
            }
        )
    return {"items": items}


# ---------------------------------------------------------------- OpenAI 兼容层（供 Open WebUI 等前端调用）
# 模型名即身份：rag-u001 → U001，rag-u002 → U002，rag-admin → admin。
# 单独用固定 API Key 校验（不参与 JWT，前端没有地方登录）。
OA_KEY = os.environ.get("RAG_OPENWEBUI_KEY", "change-me-before-use")
OA_MODELS = {"rag-u001": "U001", "rag-u002": "U002", "rag-admin": "admin"}
OA_ALIAS = {"U001": "张三 · 市场岗（员工）", "U002": "李四 · 技术岗（员工）", "admin": "管理员"}
OA_MAX_HISTORY = 6


def _oa_check_key(authorization: str) -> None:
    token = (authorization or "").strip()
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    if token != OA_KEY:
        raise HTTPException(status_code=401, detail="API Key 不正确")


def _oa_user(account: str) -> CurrentUser:
    info = load_users().get(account) or {}
    return CurrentUser(
        id=account,
        role=str(info.get("role") or "employee"),
        name=str(info.get("name") or account),
        post=str(info.get("post") or ""),
    )


def _oa_session(account: str, raw: str) -> str:
    tag = re.sub(r"[^A-Za-z0-9_-]", "", str(raw or ""))[:16] or "owui"
    return f"oa-{account}-{tag}"[:64]


def _oa_answer(question: str, history: list[dict] | None, user: CurrentUser, session_id: str) -> dict:
    """跑一遍完整 Agent 链路（输入护栏 → 记忆 → 图 → 输出护栏 → 审计），同步返回。"""
    started = time.time()
    trace_id = new_trace_id()

    guard = check_input(question)
    if guard.hit:
        log_turn(question=question, answer=BLOCK_TEXT_INPUT, user_id=user.id, mode="openwebui",
                 session_id=session_id, blocked=True, stage="pre", hit_rule=guard.rule,
                 trace_id=trace_id, elapsed=time.time() - started)
        return {"answer": BLOCK_TEXT_INPUT, "blocked": True, "reason": guard.rule,
                "trace_id": trace_id, "elapsed": round(time.time() - started, 1)}

    try:
        memories = get_long_memory().recall(user.id, question)
    except Exception:  # noqa: BLE001
        memories = []

    init: dict = {"question": question, "session_id": session_id, "user_id": user.id,
                  "memories": memories, "history": history, "steps": []}
    final = dict(init)
    for chunk in AGENT_GRAPH.stream(init, stream_mode="updates", config={"recursion_limit": 8}):
        for _node, upd in chunk.items():
            if isinstance(upd, dict):
                final.update({k: v for k, v in upd.items() if k != "steps"})

    answer = final.get("answer") or NO_ANSWER
    guard_out = check_output(answer, question, suspect=guard.suspect)
    answer_raw = None
    if guard_out.hit:
        answer_raw, answer = answer, BLOCK_TEXT_OUTPUT

    facts = _remember_turn(session_id, question, answer,
                           allow_extract=not guard_out.hit, user_id=user.id)
    log_turn(question=question, answer=answer, user_id=user.id, mode="openwebui",
             session_id=session_id, route=final.get("route"),
             hits=final.get("rag_hits"), tool_calls=final.get("tool_results"),
             memories=memories, blocked=guard_out.hit, stage=guard_out.stage,
             hit_rule=guard_out.rule, suspect=guard.suspect or guard_out.suspect,
             answer_raw=answer_raw, trace_id=trace_id, elapsed=time.time() - started)
    return {
        "answer": answer,
        "blocked": guard_out.hit,
        "reason": guard_out.rule,
        "facts": facts,
        "route": final.get("route"),
        "trace_id": trace_id,
        "elapsed": round(time.time() - started, 1),
    }


def _oa_pick(payload: dict) -> tuple[str, str, str, list[dict] | None, str]:
    """解析 OpenAI 请求：返回 (model, account, question, history, session_id)。"""
    model = str(payload.get("model") or "").lower()
    account = OA_MODELS.get(model)
    if not account:
        raise HTTPException(status_code=404, detail=f"未知模型：{model}")

    msgs = payload.get("messages") or []
    question = ""
    history: list[dict] = []
    for i in range(len(msgs) - 1, -1, -1):
        m = msgs[i]
        if m.get("role") == "user" and str(m.get("content") or "").strip():
            question = str(m["content"]).strip()
            history = [
                {"role": x.get("role"), "content": str(x.get("content") or "")[:200]}
                for x in msgs[:i]
                if x.get("role") in ("user", "assistant") and x.get("content")
            ][-OA_MAX_HISTORY:]
            break
    if not question:
        raise HTTPException(status_code=400, detail="消息为空")

    return model, account, question, (history or None), _oa_session(account, payload.get("user") or "")


def _oa_chunk(cid: str, created: int, model: str, delta: dict, finish: str | None) -> str:
    body = {
        "id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(body, ensure_ascii=False)}\n\n"


def _oa_task_reply(question: str, history: list[dict] | None) -> str | None:
    """Open WebUI 的辅助任务（生成标题/标签/追问）不送进 RAG 链路，直接秒回。"""
    if not question.startswith("### Task:"):
        return None
    head = question[:260].lower()
    if "title" in head:
        if history:
            t = str(history[0].get("content") or "").strip()
            if t:
                return t[:16]
        for mark in ("### Chat History:", "Chat History:", "USER:", "User:"):
            if mark in question:
                seg = question.split(mark, 1)[1].strip()
                line = seg.split("\n")[0].replace("USER:", "").replace("User:", "").strip()
                if line:
                    return line[:16]
        return "新对话"
    if "tag" in head:
        return ""
    if "follow" in head:
        return "[]"
    return ""


@app.get("/v1/models")
def oa_models(authorization: str = Header(default="")):
    """Open WebUI 启动时拉模型列表，这里把三个账号当成三个「模型」。"""
    _oa_check_key(authorization)
    now = int(time.time())
    return {
        "object": "list",
        "data": [
            {"id": mid, "object": "model", "created": now, "owned_by": "rag-local",
             "name": OA_ALIAS.get(acct, acct)}
            for mid, acct in OA_MODELS.items()
        ],
    }


@app.post("/v1/chat/completions")
def oa_chat(payload: dict, authorization: str = Header(default="")):
    """OpenAI 兼容对话接口：内部走完整 Agent 链路，答案原样返回。"""
    _oa_check_key(authorization)
    model, account, question, history, session_id = _oa_pick(payload)
    user = _oa_user(account)

    quick = _oa_task_reply(question, history)   # 标题/标签/追问类辅助请求，秒回不走 RAG
    if quick is not None:
        if not payload.get("stream"):
            return JSONResponse({
                "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": quick}}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            })

        def gen_quick():
            cid = f"chatcmpl-{uuid.uuid4().hex[:16]}"
            created = int(time.time())
            yield _oa_chunk(cid, created, model, {"role": "assistant", "content": ""}, None)
            yield _oa_chunk(cid, created, model, {"content": quick}, None)
            yield _oa_chunk(cid, created, model, {}, "stop")
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen_quick(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    if not payload.get("stream"):
        out = _oa_answer(question, history, user, session_id)
        return JSONResponse({
            "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": out["answer"]}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        })

    def gen():
        cid = f"chatcmpl-{uuid.uuid4().hex[:16]}"
        created = int(time.time())
        yield _oa_chunk(cid, created, model, {"role": "assistant", "content": ""}, None)
        box: queue.Queue = queue.Queue()

        def worker() -> None:
            try:
                box.put(("ok", _oa_answer(question, history, user, session_id)))
            except Exception as e:  # noqa: BLE001
                box.put(("err", f"{type(e).__name__}: {e}"))

        threading.Thread(target=worker, daemon=True).start()
        text = ""
        while True:
            try:
                kind, data = box.get(timeout=12)
            except queue.Empty:
                yield ": keepalive\n\n"      # SSE 注释行，只保活不显示
                continue
            text = f"[Agent 服务出错] {data}" if kind == "err" else data["answer"]
            break
        yield _oa_chunk(cid, created, model, {"content": text}, None)
        yield _oa_chunk(cid, created, model, {}, "stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def main() -> int:
    import uvicorn

    print("面板地址：http://127.0.0.1:8765")
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
