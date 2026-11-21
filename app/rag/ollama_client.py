"""Ollama 客户端：向量化 + 对话。只用标准库，避免额外依赖。"""
import json
import urllib.error
import urllib.request

from .config import (
    EMBED_MODEL, LLM_MODEL, LLM_NUM_CTX, LLM_NUM_PREDICT, LLM_TEMPERATURE,
    NO_THINK_TAG, OLLAMA_BASE, QUERY_INSTRUCTION,
)


class OllamaError(RuntimeError):
    pass


def _post(path: str, payload: dict, timeout: int = 120) -> dict:
    url = f"{OLLAMA_BASE}{path}"
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise OllamaError(
            f"连不上 Ollama（{OLLAMA_BASE}）。请确认 ollama 正在运行：ollama serve。原始错误：{e}"
        ) from e
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", "replace")[:300]
        except Exception:
            pass
        raise OllamaError(f"Ollama 接口 {path} 返回 {e.code}：{body}") from e


def embed_texts(texts: list[str], batch_size: int = 8) -> list[list[float]]:
    """批量向量化。优先用新版 /api/embed，失败回退旧版 /api/embeddings。"""
    if not texts:
        return []
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            resp = _post("/api/embed", {"model": EMBED_MODEL, "input": batch})
            vectors.extend(resp["embeddings"])
            continue
        except OllamaError:
            pass
        # 旧接口一次一条
        for t in batch:
            resp = _post("/api/embeddings", {"model": EMBED_MODEL, "prompt": t})
            vectors.append(resp["embedding"])
    return vectors


def embed_query(text: str) -> list[float]:
    """查询向量。bge-m3 不需要 instruction 前缀；换 v1.5 模型时由 config 打开前缀。"""
    return embed_texts([QUERY_INSTRUCTION + text])[0]


ANSWER_ANCHOR = "【答案】"


def _strip_thinking(text: str) -> str:
    """按思考结束标记截断，只留标记之后的内容。"""
    for marker in ("<｜end▁of▁thinking｜>", "</think>", "</thinking>"):
        if marker in text:
            text = text.split(marker)[-1]
    return text.strip()


def split_answer(text: str) -> tuple[bool, str]:
    """从模型输出里取出最终答案。

    坑：qwen3 会在思考过程中把「【答案】」「</think>」当成普通文本反复演练，
    所以第一次出现的标记并不可信。真正的答案永远在输出末尾，因此取**最后**一个锚点。
    """
    if ANSWER_ANCHOR in text:
        return True, text.rsplit(ANSWER_ANCHOR, 1)[1].strip()
    for marker in ("<｜end▁of▁thinking｜>", "</think>", "</thinking>"):
        if marker in text:
            return True, text.rsplit(marker, 1)[1].strip()
    return False, ""


def chat(system: str, user: str, temperature: float = LLM_TEMPERATURE, timeout: int = 1200,
         history: list[dict] | None = None) -> str:
    """单轮对话。history 为短期会话记忆里的最近几轮，按 Ollama messages 格式传入。

    超时默认 1200 秒：本机纯 CPU 跑 qwen3:4b，一个制度问题实测 400~650 秒，
    留一倍余量，避免思考没写完就被客户端掐断（表现为返回空、无法解析 JSON）。
    """
    if NO_THINK_TAG:
        user = f"{user}\n/no_think"

    messages: list[dict] = [{"role": "system", "content": system}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user})

    base_payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_ctx": LLM_NUM_CTX,
            "num_predict": LLM_NUM_PREDICT,
        },
    }
    try:
        resp = _post("/api/chat", {**base_payload, "think": False}, timeout=timeout)
    except OllamaError:
        resp = _post("/api/chat", base_payload, timeout=timeout)

    msg = resp.get("message", {})
    raw = msg.get("content") or ""
    ok, answer = split_answer(raw)
    if ok and answer:
        return answer

    thinking = (msg.get("thinking") or "").strip()
    raise OllamaError(
        "模型没有给出最终答案（思考把 num_predict 用尽、未出现结束标记或【答案】锚点）。"
        f"done_reason={resp.get('done_reason')}，输出前 80 字：{raw[:80] or thinking[:80]}"
    )


def chat_stream(system: str, user: str, timeout: int = 900, temperature: float = LLM_TEMPERATURE,
                history: list[dict] | None = None):
    """流式对话，逐块 yield ("thinking"|"content", text)。

    面板靠它把"等 5 分钟"变成"看得见的思考过程"。
    """
    if NO_THINK_TAG:
        user = f"{user}\n/no_think"

    messages: list[dict] = [{"role": "system", "content": system}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user})

    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "stream": True,
        "think": False,
        "options": {
            "temperature": temperature,
            "num_ctx": LLM_NUM_CTX,
            "num_predict": LLM_NUM_PREDICT,
        },
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/chat", data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.URLError as e:
        raise OllamaError(f"连不上 Ollama（{OLLAMA_BASE}）：{e}") from e

    with resp:
        for raw in resp:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                continue
            if obj.get("error"):
                raise OllamaError(str(obj["error"]))
            msg = obj.get("message") or {}
            if msg.get("thinking"):
                yield "thinking", msg["thinking"]
            if msg.get("content"):
                yield "content", msg["content"]
            if obj.get("done"):
                break


def health_check() -> dict:
    """返回 Ollama 可用性、模型是否就位。"""
    info = {"ok": False, "models": [], "error": None}
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE}/api/tags")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        info["models"] = [m["name"] for m in data.get("models", [])]
        info["ok"] = True
    except Exception as e:  # noqa: BLE001
        info["error"] = str(e)
    return info
