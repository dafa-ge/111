"""鉴权验收脚本：对着跑起来的面板逐条验接口权限。

用法：
    python scripts\\auth_smoke.py
    python scripts\\auth_smoke.py --base http://127.0.0.1:8765
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8765")
    args = ap.parse_args()
    base = args.base.rstrip("/")

    passed = failed = 0

    def check(label: str, cond: bool, detail: str = "") -> None:
        nonlocal passed, failed
        if cond:
            passed += 1
            print(f"  [通过] {label}")
        else:
            failed += 1
            print(f"  [失败] {label}  {detail}")

    def login(acc: str, pwd: str) -> requests.Response:
        return requests.post(f"{base}/api/auth/login",
                             json={"account": acc, "password": pwd}, timeout=15)

    print("== 1. 登录 ==")
    r = login("U001", "123456")
    check("U001 登录 200", r.status_code == 200, f"实际 {r.status_code}")
    tok_e = r.json().get("token") if r.ok else ""
    check("登录返回 token", bool(tok_e))
    check("返回员工角色", r.json()["user"]["role"] == "employee" if r.ok else False)

    r = login("admin", "admin123")
    check("admin 登录 200", r.status_code == 200, f"实际 {r.status_code}")
    tok_a = r.json().get("token") if r.ok else ""

    r = login("U001", "wrong-password")
    check("密码错 401", r.status_code == 401, f"实际 {r.status_code}")
    r = login("U999", "123456")
    check("账号不存在 401", r.status_code == 401, f"实际 {r.status_code}")

    he = {"Authorization": f"Bearer {tok_e}"}
    ha = {"Authorization": f"Bearer {tok_a}"}

    print("== 2. 未登录拦截 ==")
    for path in ["/api/status", "/api/memory/list", "/api/job"]:
        r = requests.get(f"{base}{path}", timeout=15)
        check(f"无 token {path} 401", r.status_code == 401, f"实际 {r.status_code}")
    r = requests.post(f"{base}/api/ask", json={"question": "测试"}, timeout=15)
    check("无 token /api/ask 401", r.status_code == 401, f"实际 {r.status_code}")

    print("== 3. 员工视角 ==")
    # 以管理员看到的真实文件数为基准，避免脚本硬编码数量
    baseline = len(requests.get(f"{base}/api/status", headers=ha, timeout=15).json().get("files") or [])
    r = requests.get(f"{base}/api/status", headers=he, timeout=15)
    check("U001 status 200", r.status_code == 200, f"实际 {r.status_code}")
    data = r.json()
    check("员工拿不到文件明细", "files" not in data and "files_count" in data,
          str(sorted(data)[:6]))
    check("员工能拿到文件数量", data.get("files_count") == baseline,
          f"实际 {data.get('files_count')}，基准 {baseline}")
    check("员工角色字段正确", data.get("user", {}).get("role") == "employee")
    check("员工记忆数按本人算", isinstance(data.get("memory", {}).get("count"), int))

    r = requests.get(f"{base}/api/guard", headers=he, timeout=15)
    check("员工 /api/guard 403", r.status_code == 403, f"实际 {r.status_code}")
    r = requests.get(f"{base}/api/reports", headers=he, timeout=15)
    check("员工 /api/reports 403", r.status_code == 403, f"实际 {r.status_code}")
    r = requests.post(f"{base}/api/build", headers=he, json={}, timeout=15)
    check("员工 /api/build 403", r.status_code == 403, f"实际 {r.status_code}")
    r = requests.post(f"{base}/api/docs/reingest", headers=he,
                      json={"source": "不存在.txt"}, timeout=15)
    check("员工 /api/docs/reingest 403", r.status_code == 403, f"实际 {r.status_code}")
    r = requests.post(f"{base}/api/docs/remove", headers=he,
                      json={"source": "不存在.txt"}, timeout=15)
    check("员工 /api/docs/remove 403", r.status_code == 403, f"实际 {r.status_code}")
    r = requests.post(f"{base}/api/docs/upload", headers=he, timeout=15,
                      files={"file": ("x.txt", b"hello", "text/plain")})
    check("员工 /api/docs/upload 403", r.status_code == 403, f"实际 {r.status_code}")

    print("== 4. 管理员视角 ==")
    r = requests.get(f"{base}/api/status", headers=ha, timeout=15)
    check("admin status 200", r.status_code == 200, f"实际 {r.status_code}")
    check("admin 能看文件明细", len(r.json().get("files") or []) == baseline and baseline > 0,
          f"实际 {len(r.json().get('files') or [])}")
    r = requests.get(f"{base}/api/guard", headers=ha, timeout=15)
    check("admin /api/guard 200", r.status_code == 200, f"实际 {r.status_code}")
    r = requests.get(f"{base}/api/reports", headers=ha, timeout=15)
    check("admin /api/reports 200", r.status_code == 200, f"实际 {r.status_code}")

    print("== 5. 记忆隔离 ==")
    r = requests.get(f"{base}/api/memory/list", headers=he, timeout=15)
    check("U001 读自己的记忆 200", r.status_code == 200, f"实际 {r.status_code}")
    check("U001 记忆归属正确", r.json().get("user_id") == "U001", r.json().get("user_id"))
    r = requests.get(f"{base}/api/memory/list?user_id=U002", headers=he, timeout=15)
    check("U001 读 U002 记忆 403", r.status_code == 403, f"实际 {r.status_code}")
    r = requests.get(f"{base}/api/memory/list?user_id=U001", headers=ha, timeout=15)
    check("admin 可代查 U001", r.status_code == 200, f"实际 {r.status_code}")

    print("== 6. 令牌有效性 ==")
    r = requests.get(f"{base}/api/status",
                     headers={"Authorization": "Bearer " + tok_e[:-4] + "AAAA"}, timeout=15)
    check("篡改 token 401", r.status_code == 401, f"实际 {r.status_code}")

    print(f"\n通过 {passed} 条，失败 {failed} 条")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
