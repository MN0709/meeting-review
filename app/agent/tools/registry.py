"""PRD R-P1-2：工具注册中心（s02）。

目标：新增一个能力 = 新增 1 个 handler 文件 + 1 行注册，不改主循环。
- schema 与 handler 同处一份（注册时一次传入），避免两处不一致。
- 重复注册同名工具直接报错，防止静默覆盖。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional, Union

from app.agent.tools.contract import ToolResult

# 三级权限（PRD §10.1）：readonly 放行 / write 需策略允许 / host_owned 模型永不可调用。
PermissionLevel = Literal["readonly", "write", "host_owned"]
PERMISSION_LEVELS: tuple[str, ...] = ("readonly", "write", "host_owned")

# handler 可以是同步或异步；只要求返回（或 await 出）ToolResult。
ToolHandler = Callable[..., Union[ToolResult, Awaitable[ToolResult]]]


@dataclass(frozen=True)
class ToolSpec:
    """一个工具的全部元信息：模型看 description/schema，Harness 看 permission。"""

    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: ToolHandler
    permission_level: PermissionLevel = "readonly"


class ToolRegistry:
    """工具注册表。一个进程一份即可，也可为测试单独建实例。"""

    def __init__(self) -> None:
        self._specs: Dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> ToolSpec:
        if not spec.name or not spec.name.replace("_", "").isalnum():
            raise ValueError("工具名必须是非空字母数字下划线组合：{!r}".format(spec.name))
        if spec.name in self._specs:
            raise ValueError("工具已注册，不允许重复覆盖：{}".format(spec.name))
        if spec.permission_level not in PERMISSION_LEVELS:
            raise ValueError("未知权限级别：{}".format(spec.permission_level))
        if not isinstance(spec.input_schema, dict):
            raise ValueError("input_schema 必须是 dict（JSON Schema 片段）")
        if not callable(spec.handler):
            raise ValueError("handler 必须可调用：{}".format(spec.name))
        self._specs[spec.name] = spec
        return spec

    def register_tool(
        self,
        *,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        handler: ToolHandler,
        permission_level: PermissionLevel = "readonly",
    ) -> ToolSpec:
        """便捷注册；等价于构造 ToolSpec 再 register。"""
        return self.register(
            ToolSpec(
                name=name,
                description=description,
                input_schema=input_schema,
                handler=handler,
                permission_level=permission_level,
            )
        )

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._specs.get(name)

    def list_tools(self) -> List[ToolSpec]:
        """按名称排序，保证调试端点与系统提示词输出稳定。"""
        return [self._specs[key] for key in sorted(self._specs)]

    def names(self) -> List[str]:
        return sorted(self._specs)

    def __contains__(self, name: object) -> bool:
        return name in self._specs

    def __len__(self) -> int:
        return len(self._specs)
