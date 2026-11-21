"""账号管理：初始化、改密码、新增、列出。

用法：
    python scripts\\make_user.py --init                 # 首次生成三个默认账号
    python scripts\\make_user.py --init --force         # 覆盖重建（所有密码重置为默认）
    python scripts\\make_user.py U001 新密码             # 改密码
    python scripts\\make_user.py --add U003 王小二 employee 技术岗 密码
    python scripts\\make_user.py --list
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.auth import hash_password                      # noqa: E402
from app.rag.config import USERS_JSON                  # noqa: E402

DEFAULTS = [
    ("U001", "张三", "employee", "市场岗", "123456"),
    ("U002", "李四", "employee", "技术岗", "123456"),
    ("admin", "管理员", "admin", "管理员", "admin123"),
]


def load_raw() -> dict:
    if not USERS_JSON.exists():
        return {}
    try:
        return json.loads(USERS_JSON.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print("警告：users.json 不是合法 JSON，按空文件处理", file=sys.stderr)
        return {}


def save_raw(data: dict) -> None:
    USERS_JSON.parent.mkdir(parents=True, exist_ok=True)
    # 不用 json.dump + write_text 的 BOM 问题：encoding="utf-8" 不写 BOM
    USERS_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def do_init(force: bool) -> int:
    if USERS_JSON.exists() and not force:
        print(f"账号文件已存在：{USERS_JSON}")
        print("要覆盖重建请加 --force（会重置所有密码为默认）")
        return 1
    data = {
        "_note": "演示账号。password 是 pbkdf2 哈希，改密码用 scripts\\make_user.py 账号 新密码",
    }
    for uid, name, role, post, pwd in DEFAULTS:
        data[uid] = {"name": name, "role": role, "post": post, "password": hash_password(pwd)}
    save_raw(data)
    print(f"已创建 {USERS_JSON}")
    for uid, name, role, _post, pwd in DEFAULTS:
        print(f"  {uid:<6} {name:<4} {role:<9} 初始密码 {pwd}")
    return 0


def do_set_password(account: str, pwd: str) -> int:
    data = load_raw()
    if account not in data or not isinstance(data[account], dict):
        print(f"账号不存在：{account}（先用 --add 新增）")
        return 1
    data[account]["password"] = hash_password(pwd)
    save_raw(data)
    print(f"已更新 {account} 的密码")
    return 0


def do_add(account: str, name: str, role: str, post: str, pwd: str) -> int:
    data = load_raw()
    if account in data:
        print(f"账号已存在：{account}（改密码请直接传账号和密码）")
        return 1
    if role not in {"employee", "admin"}:
        print("role 只能是 employee 或 admin")
        return 1
    data[account] = {"name": name, "role": role, "post": post, "password": hash_password(pwd)}
    save_raw(data)
    print(f"已新增 {account}（{name} / {role}）")
    return 0


def do_list() -> int:
    data = load_raw()
    rows = [(k, v) for k, v in data.items() if isinstance(v, dict)]
    if not rows:
        print("还没有账号，先跑 --init")
        return 1
    print(f"{'账号':<8}{'姓名':<8}{'角色':<10}{'岗位'}")
    for uid, info in rows:
        print(f"{uid:<8}{info.get('name',''):<8}{info.get('role',''):<10}{info.get('post','')}")
    return 0


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="账号管理")
    parser.add_argument("account", nargs="?", help="账号（配合密码改密）")
    parser.add_argument("password", nargs="?", help="新密码")
    parser.add_argument("--init", action="store_true", help="生成默认账号")
    parser.add_argument("--force", action="store_true", help="覆盖已有文件")
    parser.add_argument("--list", action="store_true", help="列出账号")
    parser.add_argument("--add", nargs=5, metavar=("账号", "姓名", "角色", "岗位", "密码"),
                        help="新增账号")
    args = parser.parse_args()

    if args.init:
        return do_init(args.force)
    if args.list:
        return do_list()
    if args.add:
        return do_add(*args.add)
    if args.account and args.password:
        return do_set_password(args.account, args.password)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
