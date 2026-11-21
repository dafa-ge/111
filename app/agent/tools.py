"""两个内置工具：年假查询、差旅标准校验。

纯函数，毫秒级，不依赖模型，可以单独单测。
schema 同时给了 OpenAI / Ollama 的 function calling 格式（两者结构一致）。
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]      # RAG知识库\
USER_FILE = ROOT / "mock_data" / "user_info.json"
RULE_FILE = ROOT / "rule_data.json"

ANNUAL_RULE = "满 1 年不满 10 年 5 天／满 10 年不满 20 年 10 天／满 20 年 15 天"


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _expected_annual(work_years: int) -> int:
    """按《考勤、加班、休假管理办法》由工龄推算应享年假。"""
    if work_years < 1:
        return 0
    if work_years < 10:
        return 5
    if work_years < 20:
        return 10
    return 15


def query_annual_leave(user_id: str = "default") -> dict:
    """查某员工当年年假与累计工龄。

    user_id 传 default / 空 时用档案里的 _default_user（面板没有登录态）。
    """
    data = _load(USER_FILE)
    uid = data.get("_default_user", "U001") if user_id in ("", "default", None) else user_id
    rec = data.get(uid)
    if not rec:
        return {"ok": False, "error": f"没有工号 {user_id} 的档案", "user_id": user_id}

    work_years = rec["work_years"]
    total = rec["annual_total"]
    used = rec["annual_used"]
    left = rec.get("annual_left", total - used)
    out = {
        "ok": True,
        "user_id": uid,
        "name": rec["name"],
        "post": rec["post"],
        "work_years": work_years,
        "annual_total": total,
        "annual_used": used,
        "annual_left": left,
        "rule": ANNUAL_RULE,
        "source": "考勤、加班、休假管理办法.txt · 年假规则",
    }
    expect = _expected_annual(work_years)
    if expect != total:
        out["warning"] = (f"档案额度 {total} 天与工龄 {work_years} 年应享 {expect} 天不一致，"
                          f"请人事核对")
    return out


def check_travel_standard(post: str, scope: str | None = None) -> dict:
    """查某岗位在省内／省外的住宿、餐补、交通标准。

    scope 缺省时**不猜**，返回 need_clarify，由上层反问用户。
    """
    data = _load(RULE_FILE)
    std = data["standards"].get(post)
    if std is None:
        return {"ok": False,
                "error": f"岗位「{post}」不在标准表里，可用：{'、'.join(data['standards'])}"}

    if not scope:
        return {
            "ok": False,
            "need_clarify": True,
            "post": post,
            "question": f"{post}的住宿标准分省内、省外两档，你这次出差是省内还是省外？",
            "available_scope": list(std.keys()),
            "source": data["source"],
        }

    item = std.get(scope)
    if item is None:
        return {"ok": False, "error": f"出差范围「{scope}」不合法，只能是 省内／省外"}
    return {"ok": True, "post": post, "scope": scope, **item, "source": data["source"]}


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "query_annual_leave",
            "description": ("查询员工当年年假：剩余天数、已用天数、总额、累计工龄。"
                            "用户问“我还有几天年假”“我工龄几年”“年假用了几天的”时使用。"),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string",
                                "description": "员工工号（如 U001）；不知道就传 default"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_travel_standard",
            "description": ("查询某岗位在省内或省外的出差住宿上限、餐补标准、交通标准。"
                            "用户问“我出差住宿能报多少”“餐补多少”“一等座能不能坐”时使用。"
                            "如果用户没说省内还是省外，不要猜，把 scope 留空。"),
            "parameters": {
                "type": "object",
                "properties": {
                    "post": {"type": "string", "enum": ["职能岗", "技术岗", "市场岗"],
                             "description": "岗位；从用户信息里取，没给就问"},
                    "scope": {"type": "string", "enum": ["省内", "省外"],
                              "description": "出差范围；用户没说就不要填"},
                },
                "required": ["post"],
            },
        },
    },
]

TOOL_FUNCS = {
    "query_annual_leave": query_annual_leave,
    "check_travel_standard": check_travel_standard,
}


def call_tool(name: str, args: dict | None = None) -> dict:
    """统一入口：任何异常都转成 {"ok": false, "error": ...}，不让工具炸掉整轮问答。"""
    fn = TOOL_FUNCS.get(name)
    if fn is None:
        return {"ok": False, "error": f"未知工具 {name}"}
    try:
        return fn(**(args or {}))
    except TypeError as e:
        return {"ok": False, "error": f"参数不对：{e}"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
