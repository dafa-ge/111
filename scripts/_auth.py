"""smoke 脚本共用的登录态。

用法（在每个脚本的 import 之后）：
    from _auth import headers as _auth_headers
    requests = requests.Session()
    requests.headers.update(_auth_headers("admin"))      # 或 "U001" / "U002"

这样脚本里原有的 requests.get/post 一行都不用改。
"""
from __future__ import annotations

import requests

BASE = "http://127.0.0.1:8765"
DEFAULT_PWD = {"U001": "123456", "U002": "123456", "admin": "admin123"}

_cache: dict[tuple[str, str], str] = {}


def token(account: str = "admin", base: str = BASE) -> str:
    key = (base, account)
    if key in _cache:
        return _cache[key]
    r = requests.post(
        f"{base}/api/auth/login",
        json={"account": account, "password": DEFAULT_PWD[account]},
        timeout=15,
    )
    r.raise_for_status()
    _cache[key] = r.json()["token"]
    return _cache[key]


def headers(account: str = "admin", base: str = BASE) -> dict[str, str]:
    return {"Authorization": f"Bearer {token(account, base)}"}
