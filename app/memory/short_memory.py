"""短期会话记忆：同一 session_id 维持最近若干轮对话。

等价于 LangChain ConversationBufferMemory 的窗口版（该类在 0.3 起已废弃，
官方替代是 ChatMessageHistory，本实现接口语义与之对齐：
append ≈ save_context，history ≈ load_memory_variables，clear ≈ clear）。

存储：进程内 deque + 每会话一个 json 落盘（面板重启后还能接上文）。
"""
from __future__ import annotations

import json
import re
import threading
import time
from collections import deque
from pathlib import Path

from ..rag.config import SESSION_MAX_MESSAGES, SESSION_TTL_DAYS, SESSIONS_DIR

# session_id 来自前端，必须白名单校验，防止拼路径穿越
SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{6,64}$")


class ShortMemory:
    def __init__(
        self,
        max_messages: int = SESSION_MAX_MESSAGES,
        persist_dir: Path | str | None = SESSIONS_DIR,
        ttl_days: int = SESSION_TTL_DAYS,
    ):
        self.max_messages = max_messages
        self.dir = Path(persist_dir) if persist_dir else None
        self.ttl_days = ttl_days
        self._sessions: dict[str, deque] = {}
        self._lock = threading.Lock()
        if self.dir:
            self._load_all()

    # ---------------------------------------------------------------- 校验
    @staticmethod
    def valid_id(session_id: str) -> bool:
        return bool(session_id and SESSION_ID_RE.match(session_id))

    # ---------------------------------------------------------------- 落盘
    def _path(self, session_id: str) -> Path:
        return self.dir / f"{session_id}.json"

    def _load_all(self) -> None:
        if not self.dir.exists():
            return
        cutoff = time.time() - self.ttl_days * 86400
        for p in self.dir.glob("*.json"):
            try:
                if p.stat().st_mtime < cutoff:
                    p.unlink()
                    continue
                data = json.loads(p.read_text(encoding="utf-8"))
                sid = data.get("session_id") or p.stem
                if not self.valid_id(sid):
                    continue
                self._sessions[sid] = deque(
                    data.get("messages", []), maxlen=self.max_messages
                )
            except Exception:  # noqa: BLE001
                continue

    def _save(self, session_id: str) -> None:
        if not self.dir:
            return
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "session_id": session_id,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "messages": list(self._sessions.get(session_id, [])),
            }
            target = self._path(session_id)
            tmp = target.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            tmp.replace(target)
        except Exception:  # noqa: BLE001
            pass  # 落盘失败不影响问答

    # ---------------------------------------------------------------- 读写
    def append(self, session_id: str, role: str, content: str) -> None:
        if not self.valid_id(session_id):
            raise ValueError(f"非法 session_id：{session_id}")
        with self._lock:
            q = self._sessions.setdefault(session_id, deque(maxlen=self.max_messages))
            q.append({"role": role, "content": content, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})
        self._save(session_id)

    def history(self, session_id: str) -> list[dict]:
        if not self.valid_id(session_id):
            return []
        with self._lock:
            return list(self._sessions.get(session_id, []))

    def messages_for_llm(self, session_id: str) -> list[dict]:
        """只保留 role/content，供 Ollama messages 使用。"""
        return [{"role": m["role"], "content": m["content"]} for m in self.history(session_id)]

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
        if self.dir and self._path(session_id).exists():
            try:
                self._path(session_id).unlink()
            except Exception:  # noqa: BLE001
                pass

    def list_sessions(self) -> list[dict]:
        with self._lock:
            return [
                {"session_id": sid, "messages": len(q),
                 "last": q[-1]["ts"] if q else None}
                for sid, q in self._sessions.items()
            ]


_short: ShortMemory | None = None


def get_short_memory() -> ShortMemory:
    global _short
    if _short is None:
        _short = ShortMemory()
    return _short
