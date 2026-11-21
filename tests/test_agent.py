"""单测：两个工具的行为 + 参数缺失。不调模型，毫秒级。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.tools import call_tool, check_travel_standard, query_annual_leave  # noqa: E402


class TestAnnualLeave(unittest.TestCase):
    def test_default_user(self):
        r = query_annual_leave()
        self.assertTrue(r["ok"])
        self.assertEqual(r["user_id"], "U001")
        self.assertEqual(r["annual_left"], 3)
        self.assertEqual(r["work_years"], 4)

    def test_explicit_user(self):
        r = query_annual_leave("U002")
        self.assertEqual(r["annual_left"], 4)
        self.assertEqual(r["work_years"], 12)

    def test_unknown_user(self):
        r = query_annual_leave("U999")
        self.assertFalse(r["ok"])
        self.assertIn("error", r)


class TestTravelStandard(unittest.TestCase):
    def test_in_province(self):
        r = check_travel_standard("市场岗", "省内")
        self.assertTrue(r["ok"])
        self.assertEqual(r["hotel"], 350)
        self.assertEqual(r["meal"], 100)

    def test_out_province(self):
        r = check_travel_standard("技术岗", "省外")
        self.assertEqual(r["hotel"], 380)

    def test_need_clarify(self):
        """没说省内省外 -> 反问，不猜。"""
        r = check_travel_standard("市场岗")
        self.assertFalse(r["ok"])
        self.assertTrue(r["need_clarify"])
        self.assertIn("省内还是省外", r["question"])

    def test_bad_post(self):
        r = check_travel_standard("销售岗", "省内")
        self.assertFalse(r["ok"])


class TestCallTool(unittest.TestCase):
    def test_dispatch(self):
        r = call_tool("query_annual_leave", {"user_id": "default"})
        self.assertTrue(r["ok"])

    def test_unknown_tool(self):
        r = call_tool("nope", {})
        self.assertFalse(r["ok"])

    def test_bad_args(self):
        r = call_tool("check_travel_standard", {"foo": "bar"})
        self.assertFalse(r["ok"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
