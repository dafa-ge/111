"""JWT 鉴权与角色权限。

两层分工：
    AuthMiddleware              解析 Authorization: Bearer <token>，把用户挂到 request.state.user；
                                白名单路径（/、/api/auth/login）不校验。解析失败统一 401。
    require_user / require_admin 在接口签名上声明所需角色，不满足返回 403。

密码用标准库 pbkdf2_hmac（10 万轮），不引入 bcrypt；令牌优先用 PyJWT，
导入失败时自动退化为标准库实现的同格式 HS256（避免因装不上库而卡住）。
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass

from fastapi import Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from .rag.config import AUTH_ENABLED, JWT_ALG, JWT_SECRET_FILE, JWT_TTL_HOURS, USERS_JSON

try:
    import jwt as _pyjwt          # PyJWT
except ImportError:                # pragma: no cover
    _pyjwt = None

HAS_PYJWT = _pyjwt is not None

# 不用登录就能访问的路径
WHITELIST = ("/", "/api/auth/login", "/favicon.ico")
PBKDF2_ITER = 100_000


# ---------------------------------------------------------------- 用户
@dataclass(frozen=True)
class CurrentUser:
    id: str
    role: str
    name: str
    post: str

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    def as_dict(self) -> dict:
        return {"id": self.id, "role": self.role, "name": self.name, "post": self.post}


# 鉴权关闭时用的虚拟身份（回到改造前：本机、全权限）
ANON = CurrentUser(id="default", role="admin", name="本机", post="管理员")


# ---------------------------------------------------------------- 密钥
_secret_cache: str | None = None


def secret() -> str:
    """JWT 签名密钥：首次调用生成并落盘，之后一直复用（重启后旧 token 仍有效）。"""
    global _secret_cache
    if _secret_cache:
        return _secret_cache
    JWT_SECRET_FILE.parent.mkdir(parents=True, exist_ok=True)
    if JWT_SECRET_FILE.exists():
        _secret_cache = JWT_SECRET_FILE.read_text(encoding="utf-8").strip()
        if _secret_cache:
            return _secret_cache
    _secret_cache = secrets.token_urlsafe(48)
    JWT_SECRET_FILE.write_text(_secret_cache, encoding="utf-8")
    return _secret_cache


# ---------------------------------------------------------------- 密码
def hash_password(pwd: str, salt: str | None = None) -> str:
    """格式：pbkdf2$<轮数>$<salt_hex>$<hash_hex>"""
    salt = salt or secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", pwd.encode("utf-8"), bytes.fromhex(salt), PBKDF2_ITER)
    return f"pbkdf2${PBKDF2_ITER}${salt}${dk.hex()}"


def verify_password(pwd: str, stored: str) -> bool:
    try:
        algo, iters, salt, digest = (stored or "").split("$")
        if algo != "pbkdf2":
            return False
        dk = hashlib.pbkdf2_hmac("sha256", pwd.encode("utf-8"), bytes.fromhex(salt), int(iters))
        return hmac.compare_digest(dk.hex(), digest)
    except Exception:  # noqa: BLE001  格式坏 = 验证失败，不抛
        return False


# ---------------------------------------------------------------- 账号
_users_cache: dict[str, object] = {"mtime": None, "data": {}}


def load_users() -> dict[str, dict]:
    """按 mtime 缓存重读，改了 users.json 不用重启。"""
    try:
        mtime = USERS_JSON.stat().st_mtime
    except OSError:
        return {}
    if _users_cache["mtime"] == mtime:
        return _users_cache["data"]  # type: ignore[return-value]
    try:
        raw = json.loads(USERS_JSON.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        raw = {}
    data = {k: v for k, v in raw.items() if not k.startswith("_") and isinstance(v, dict)}
    _users_cache.update({"mtime": mtime, "data": data})
    return data


def authenticate(account: str, password: str) -> CurrentUser | None:
    """账号或密码错都返回 None——调用方统一回"账号或密码不正确"，不泄露账号是否存在。"""
    info = load_users().get((account or "").strip())
    if not info:
        return None
    if not verify_password(password or "", info.get("password", "")):
        return None
    return CurrentUser(
        id=(account or "").strip(),
        role=str(info.get("role") or "employee"),
        name=str(info.get("name") or account),
        post=str(info.get("post") or ""),
    )


# ---------------------------------------------------------------- 令牌
def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64d(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def issue_token(user: CurrentUser) -> str:
    now = int(time.time())
    payload = {
        "sub": user.id, "role": user.role, "name": user.name, "post": user.post,
        "iat": now, "exp": now + JWT_TTL_HOURS * 3600,
    }
    if _pyjwt:
        return _pyjwt.encode(payload, secret(), algorithm=JWT_ALG)
    # 退化实现：标准库自签同格式 HS256 JWT
    header = {"alg": JWT_ALG, "typ": "JWT"}
    seg = (_b64e(json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
           + "." + _b64e(json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")))
    sig = hmac.new(secret().encode("utf-8"), seg.encode("ascii"), hashlib.sha256).digest()
    return seg + "." + _b64e(sig)


def decode_token(token: str) -> dict:
    if _pyjwt:
        return _pyjwt.decode(token, secret(), algorithms=[JWT_ALG])
    parts = (token or "").split(".")
    if len(parts) != 3:
        raise ValueError("token 格式错误")
    seg = parts[0] + "." + parts[1]
    sig = hmac.new(secret().encode("utf-8"), seg.encode("ascii"), hashlib.sha256).digest()
    if not hmac.compare_digest(_b64e(sig), parts[2]):
        raise ValueError("签名不匹配")
    payload = json.loads(_b64d(parts[1]))
    if int(payload.get("exp") or 0) < int(time.time()):
        raise ValueError("token 已过期")
    return payload


def user_from_token(token: str) -> CurrentUser:
    payload = decode_token(token)
    sub = payload.get("sub")
    if not sub:
        raise ValueError("token 缺 sub")
    info = load_users().get(str(sub)) or {}
    return CurrentUser(
        id=str(sub),
        role=str(payload.get("role") or info.get("role") or "employee"),
        name=str(payload.get("name") or info.get("name") or sub),
        post=str(payload.get("post") or info.get("post") or ""),
    )


def login(account: str, password: str, minutes: int | None = None) -> dict | None:
    """成功返回 {token, expires_in, user}，失败返回 None。"""
    user = authenticate(account, password)
    if not user:
        return None
    token = issue_token(user)
    return {
        "token": token,
        "token_type": "Bearer",
        "expires_in": (minutes or int(JWT_TTL_HOURS * 60)),
        "user": user.as_dict(),
    }


# ---------------------------------------------------------------- 中间件
def _unauthorized() -> JSONResponse:
    # 文案固定，不区分"没带 token / 签名错 / 过期"，避免给探测者信息
    return JSONResponse(status_code=401, content={"detail": "未登录或登录已过期"})


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        request.state.user = ANON
        if not AUTH_ENABLED:
            return await call_next(request)
        if request.url.path in WHITELIST or request.url.path.startswith("/v1/"):
            # /v1/* 是 OpenAI 兼容层，由它自己用固定 API Key 校验（前端没地方登录）
            return await call_next(request)
        header = request.headers.get("authorization") or ""
        if not header.lower().startswith("bearer "):
            return _unauthorized()
        try:
            request.state.user = user_from_token(header[7:].strip())
        except Exception:  # noqa: BLE001  签名错/过期/格式错一律 401
            return _unauthorized()
        return await call_next(request)


# ---------------------------------------------------------------- 依赖
def require_user(request: Request) -> CurrentUser:
    user = getattr(request.state, "user", None)
    if user is None:
        raise HTTPException(status_code=401, detail="未登录或登录已过期")
    return user


def require_admin(user: CurrentUser = Depends(require_user)) -> CurrentUser:
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="权限不足，需要管理员")
    return user
