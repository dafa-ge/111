"""单任务队列：入库、重建这类会改向量库的操作，同一时刻只允许一个。"""
from __future__ import annotations

import threading
import traceback
from datetime import datetime


class JobRunner:
    def __init__(self):
        self._run_lock = threading.Lock()      # 保证串行
        self._state_lock = threading.Lock()    # 保护状态读写
        self._state = {"status": "idle", "label": "", "started_at": None,
                       "ended_at": None, "result": None, "error": None}

    def is_busy(self) -> bool:
        return self._run_lock.locked()

    def submit(self, label: str, fn) -> bool:
        """返回 False 表示已有任务在跑，调用方应提示稍后再试。"""
        if not self._run_lock.acquire(blocking=False):
            return False
        self._patch({"status": "running", "label": label,
                     "started_at": datetime.now().strftime("%H:%M:%S"),
                     "ended_at": None, "result": None, "error": None})
        threading.Thread(target=self._run, args=(label, fn), daemon=True).start()
        return True

    def _run(self, label: str, fn) -> None:
        try:
            result = fn()
            self._patch({"status": "done", "result": _compact(result),
                         "ended_at": datetime.now().strftime("%H:%M:%S")})
        except Exception as e:  # noqa: BLE001
            self._patch({"status": "failed", "error": f"{type(e).__name__}: {e}",
                         "ended_at": datetime.now().strftime("%H:%M:%S"),
                         "trace": traceback.format_exc()[-400:]})
        finally:
            self._run_lock.release()

    def _patch(self, patch: dict) -> None:
        with self._state_lock:
            self._state.update(patch)

    def snapshot(self) -> dict:
        with self._state_lock:
            return dict(self._state)


def _compact(result):
    if not isinstance(result, dict):
        return result
    keys = ("new", "updated", "skipped", "removed", "empty_files", "errors",
            "chunks_in_file", "chunks_in_chroma", "consistent")
    return {k: result.get(k) for k in keys if k in result}
