"""面板接口冒烟测试：状态、问答（含拒答与 SSE）、上传、入库、移出、非法输入。

用法：先启动面板，再执行
    python scripts/panel_smoke.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _auth import headers as _auth_headers      # noqa: E402

# 加鉴权后所有接口都要 token：这里换成已登录的 Session，下面的调用不用逐条改
requests = requests.Session()                   # noqa: PLW0127
requests.headers.update(_auth_headers("admin"))

BASE = "http://127.0.0.1:8765"
DOCS = Path(__file__).resolve().parents[1] / "demo_docs"
TMP_NAME = "面板冒烟测试_制度_20261118_v1.0.txt"
TMP_TEXT = (
    "面板冒烟测试文件，仅用于验证上传与入库链路，可删除。\n\n"
    "十一、测试条款：此文件内容仅用于验证面板上传、体检、入库三个环节是否正常。"
)


def show(tag, resp):
    try:
        body = resp.json()
    except Exception:  # noqa: BLE001
        body = resp.text[:200]
    text = json.dumps(body, ensure_ascii=False)
    print(f"[{tag}] HTTP {resp.status_code} {text[:600]}")
    return resp.status_code, body


def wait_job(timeout=90):
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = requests.get(f"{BASE}/api/job", timeout=10).json()
        if st.get("status") in {"done", "failed", "idle"}:
            return st
        time.sleep(2)
    return {"status": "timeout"}


def cleanup():
    r = requests.post(f"{BASE}/api/docs/remove", json={"source": TMP_NAME}, timeout=30)
    if r.ok:
        wait_job(40)
    p = DOCS / TMP_NAME
    if p.exists():
        p.unlink()
        print(f"[cleanup] 已删除 {TMP_NAME}")


def main() -> int:
    print("=" * 60)
    ok = True

    r = requests.get(f"{BASE}/api/status", timeout=40)
    code, body = show("status", r)
    ok &= code == 200 and body.get("consistent") is True

    r = requests.get(f"{BASE}/", timeout=20)
    show("index", r)
    ok &= r.status_code == 200 and "RAG" in r.text

    # 拒答路径：不调 LLM，秒回
    r = requests.post(f"{BASE}/api/ask", json={"question": "食堂几点开饭？"}, timeout=60)
    code, body = show("ask-拒答", r)
    ok &= code == 200 and body.get("confident") is False
    ok &= "未找到" in (body.get("answer") or "")

    # 空问题
    r = requests.post(f"{BASE}/api/ask", json={"question": "   "}, timeout=20)
    show("ask-空问题", r)
    ok &= r.status_code == 400

    # SSE 拒答路径
    print("[sse] 拒答流 …")
    with requests.post(f"{BASE}/api/ask/stream", json={"question": "公司有没有股票期权？"},
                       stream=True, timeout=60) as resp:
        events = []
        for line in resp.iter_lines(decode_unicode=True):
            if line:
                events.append(line)
                print("   ", line[:160])
        kinds = [e.split(":", 1)[1].strip() for e in events if e.startswith("event:")]
        ok &= "sources" in kinds and "done" in kinds

    # 上传真实文件并等待入库
    cleanup()
    files = {"file": (TMP_NAME, TMP_TEXT.encode("utf-8"), "text/plain")}
    r = requests.post(f"{BASE}/api/docs/upload", files=files, data={"overwrite": "false"}, timeout=60)
    code, body = show("upload", r)
    ok &= code == 200 and body.get("ok") is True

    job = wait_job()
    print(f"[job] {json.dumps(job, ensure_ascii=False)[:400]}")
    ok &= job.get("status") == "done"
    res = job.get("result") or {}
    ok &= len(res.get("new") or []) >= 1

    r = requests.get(f"{BASE}/api/status", timeout=40)
    _, body = show("status-after", r)
    ok &= body.get("consistent") is True

    # 上传后能检索到新内容（只读检索结果，不跑 LLM，避免等 5 分钟）
    hits: list[dict] = []
    with requests.post(f"{BASE}/api/ask/stream",
                       json={"question": "面板冒烟测试文件是干什么的？"},
                       stream=True, timeout=60) as resp:
        for line in resp.iter_lines(decode_unicode=True):
            if line and line.startswith("data:"):
                try:
                    obj = json.loads(line[5:].strip())
                except Exception:  # noqa: BLE001
                    continue
                if obj.get("hits"):
                    hits = obj["hits"]
                    break
    first = hits[0]["source"] if hits else "-"
    print(f"[retrieve-new] 命中 {len(hits)} 条，首条 = {first}")
    ok &= bool(hits)

    # 空文件
    r = requests.post(f"{BASE}/api/docs/upload",
                      files={"file": ("空文件_制度_20261118_v1.0.txt", b"", "text/plain")},
                      data={"overwrite": "false"}, timeout=30)
    show("upload-空文件", r)
    ok &= r.status_code == 400

    # 非法扩展名
    r = requests.post(f"{BASE}/api/docs/upload",
                      files={"file": ("bad.exe", b"MZ", "application/octet-stream")},
                      data={"overwrite": "false"}, timeout=30)
    show("upload-非法扩展", r)
    ok &= r.status_code == 415

    # 路径穿越
    r = requests.post(f"{BASE}/api/docs/upload",
                      files={"file": ("../../evil.txt", b"x", "text/plain")},
                      data={"overwrite": "false"}, timeout=30)
    show("upload-路径穿越", r)
    ok &= r.status_code == 400
    leaked = (DOCS.parent / "evil.txt").exists() or (DOCS / "evil.txt").exists()
    print(f"[path] 出现 evil.txt = {leaked}")
    ok &= not leaked

    cleanup()
    job = wait_job()
    print(f"[cleanup-job] {job.get('status')}")

    print("=" * 60)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
