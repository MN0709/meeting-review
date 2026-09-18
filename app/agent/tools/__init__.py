"""Agent 工具层。契约见 contract.py，注册中心见 registry.py。

`default_registry` 是进程级默认注册表：工具在各自模块里向它注册。
`pipeline` 模式下即使注册了工具也不会被调用（AGENT_MODE 开关）。
"""

from app.agent.tools.contract import ToolResult, failure, success
from app.agent.tools.registry import ToolRegistry, ToolSpec

default_registry = ToolRegistry()

__all__ = ["ToolResult", "success", "failure", "ToolRegistry", "ToolSpec", "default_registry"]
