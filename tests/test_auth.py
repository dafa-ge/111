"""鉴权单测：密码哈希、令牌签发/校验、篡改与过期、账号登录。

不依赖面板进程，毫秒级。
"""
from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jwt as pyjwt                                     # noqa: E402

from app import auth                                    # noqa: E402
from app.rag.config import USERS_JSON                   # noqa: E402


class TestPassword(unittest.TestCase):
    def test_hash_and_verify(self) -> None:
        h = auth.hash_password("123456")
        self.assertTrue(h.startswith("pbkdf2$100000$"))
        self.assertTrue(auth.verify_password("123456", h))
        self.assertFalse(auth.verify_password("wrong", h))

    def test_same_password_different_salt(self) -> None:
        a = auth.hash_password("123456")
        b = auth.hash_password("123456")
        self.assertNotEqual(a, b, "盐必须随机，否则同密码同哈希")

    def test_broken_hash_is_false(self) -> None:
        for bad in ["", "plain", "pbkdf2$x$y", "md5$1$aa$bb"]:
            self.assertFalse(auth.verify_password("123456", bad))


class TestToken(unittest.TestCase):
    def setUp(self) -> None:
        self.user = auth.CurrentUser("U001", "employee", "张三", "市场岗")

    def test_roundtrip(self) -> None:
        token = auth.issue_token(self.user)
        got = auth.user_from_token(token)
        self.assertEqual(got.id, "U001")
        self.assertEqual(got.role, "employee")
        self.assertEqual(got.name, "张三")

    def test_tampered_token(self) -> None:
        token = auth.issue_token(self.user)
        bad = token[:-4] + ("aaaa" if token[-4:] != "aaaa" else "bbbb")
        with self.assertRaises(Exception):
            auth.user_from_token(bad)

    def test_expired_token(self) -> None:
        payload = {"sub": "U001", "role": "employee", "exp": int(time.time()) - 10}
        token = pyjwt.encode(payload, auth.secret(), algorithm="HS256")
        with self.assertRaises(Exception):
            auth.user_from_token(token)

    def test_garbage_token(self) -> None:
        for bad in ["", "abc", "a.b.c", "Bearer x"]:
            with self.assertRaises(Exception):
                auth.user_from_token(bad)


class TestAccount(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not USERS_JSON.exists():
            raise unittest.SkipTest(f"缺账号文件 {USERS_JSON}，先跑 make_user.py --init")

    def test_admin_login(self) -> None:
        res = auth.login("admin", "admin123")
        self.assertIsNotNone(res)
        self.assertEqual(res["user"]["role"], "admin")
        self.assertTrue(auth.user_from_token(res["token"]).is_admin)

    def test_employee_login(self) -> None:
        res = auth.login("U001", "123456")
        self.assertIsNotNone(res)
        self.assertEqual(res["user"]["role"], "employee")
        self.assertEqual(res["user"]["post"], "市场岗")

    def test_wrong_password(self) -> None:
        self.assertIsNone(auth.login("U001", "wrong"))

    def test_unknown_account(self) -> None:
        self.assertIsNone(auth.login("U999", "123456"))

    def test_admin_is_admin_flag(self) -> None:
        self.assertTrue(auth.CurrentUser("admin", "admin", "管理员", "管理员").is_admin)
        self.assertFalse(auth.CurrentUser("U001", "employee", "张三", "市场岗").is_admin)


if __name__ == "__main__":
    unittest.main(verbosity=2)
