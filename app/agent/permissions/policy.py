"""PRD R-P1-5 / §10.1：三级权限模型（readonly / write / host_owned）。

这是确定性的代码判定，模型无法绕过：
- readonly：默认放行；
- write：仅当写工具总开关打开且工具在白名单内；
- host_owned：**永远拒绝**（删会议/删项目/删声纹/合并成员/写长期声纹）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Optional, Set

DECISION_ALLOW = "allow"
DECISION_DENY_POLICY = "deny_policy"
DECISION_DENY_HOST_OWNED = "deny_host_owned"

# PRD §9.4：只有人能执行、模型永远不可调用的操作。
HOST_OWNED_TOOLS: FrozenSet[str] = frozenset(
    {
        "delete_meeting",
        "delete_project",
        "delete_voiceprint",
        "merge_members",
        "finalize_voiceprints",
    }
)


@dataclass(frozen=True)
class PermissionDecision:
    allow: bool
    decision: str  # allow / deny_policy / deny_host_owned（写入 agent_audit）
    code: str      # 拒绝时对应 ToolResult.code
    reason: str    # 中文可读原因（回灌给模型）


class PermissionPolicy:
    """按工具权限级别放行或拒绝。纯函数式判定，不接触业务数据。"""

    def __init__(
        self,
        *,
        tools_enabled: str = "readonly",
        write_tools_enabled: bool = False,
        host_owned: FrozenSet[str] = HOST_OWNED_TOOLS,
    ) -> None:
        self.write_tools_enabled = bool(write_tools_enabled)
        self.host_owned = frozenset(host_owned)
        raw = (tools_enabled or "").strip()
        if raw in ("", "readonly"):
            # 放行全部 readonly 工具。
            self.allow_all_readonly = True
            self.allowlist: Set[str] = set()
        else:
            self.allow_all_readonly = False
            self.allowlist = {item.strip() for item in raw.split(",") if item.strip()}

    def decide(self, tool_name: str, permission_level: Optional[str]) -> PermissionDecision:
        if tool_name in self.host_owned:
            return PermissionDecision(
                False, DECISION_DENY_HOST_OWNED, "denied_host_owned",
                "删除/合并/写入长期声纹等操作只能由人在界面上执行，你无权调用。",
            )
        if permission_level is None:
            return PermissionDecision(
                False, DECISION_DENY_POLICY, "denied_policy", "未知工具，未在允许范围内。"
            )
        if permission_level == "host_owned":
            return PermissionDecision(
                False, DECISION_DENY_HOST_OWNED, "denied_host_owned",
                "该操作属于 host-owned，模型不可调用。",
            )
        if permission_level == "readonly":
            if self.allow_all_readonly or tool_name in self.allowlist:
                return PermissionDecision(True, DECISION_ALLOW, "ok", "")
            return PermissionDecision(
                False, DECISION_DENY_POLICY, "denied_policy", "该工具不在当前允许清单内。"
            )
        if permission_level == "write":
            if self.write_tools_enabled and tool_name in self.allowlist:
                return PermissionDecision(True, DECISION_ALLOW, "ok", "")
            return PermissionDecision(
                False, DECISION_DENY_POLICY, "denied_policy", "写工具当前未开启，请不要尝试。"
            )
        return PermissionDecision(
            False, DECISION_DENY_POLICY, "denied_policy", "未知权限级别。"
        )
