"""Configuration loading utilities for the agentic framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping

try:  # pragma: no cover - optional dependency
    import yaml
except ModuleNotFoundError:  # pragma: no cover - fallback parser is exercised in tests
    yaml = None


@dataclass(frozen=True)
class LLMConfig:
    """Metadata for the large language model backing the agent."""

    provider: str
    model: str
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentSettings:
    """Agent-specific configuration derived from the YAML file."""

    name: str
    llm: LLMConfig
    tools: Iterable[str]
    plan: List[Mapping[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class ToolSettings:
    """Definition for a tool registered in the framework."""

    class_path: str
    args: Mapping[str, Any] = field(default_factory=dict)


def load_config(config_file: Any) -> Dict[str, Any]:
    """Load a YAML configuration file and return the parsed dictionary."""

    path = Path(config_file)
    if not path.exists():
        raise FileNotFoundError(f"Config file '{path}' does not exist")

    with path.open("r", encoding="utf-8") as handle:
        text = handle.read()

    if yaml is not None:  # pragma: no cover - exercised when PyYAML is available
        data = yaml.safe_load(text) or {}
    else:
        data = _load_simple_yaml(text)

    if not isinstance(data, MutableMapping):
        raise ValueError("Configuration must be a mapping at the top level")
    return dict(data)


def _load_simple_yaml(text: str) -> Dict[str, Any]:
    """Parse a minimal subset of YAML supporting mappings and sequences."""

    lines = [line.rstrip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
    processed: List[str] = []
    for line in lines:
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if stripped.startswith("- ") and ":" in stripped[2:]:
            processed.append(" " * indent + "-")
            processed.append(" " * (indent + 2) + stripped[2:])
        else:
            processed.append(line)
    return _parse_block(processed, 0, 0)[0]


def _parse_block(lines: List[str], indent: int, index: int) -> Any:
    container: Any = None
    items_list: List[Any] = []
    items_map: Dict[str, Any] = {}

    while index < len(lines):
        line = lines[index]
        stripped = line.lstrip()
        current_indent = len(line) - len(stripped)
        if current_indent < indent:
            break
        if current_indent > indent:
            raise ValueError(f"Unexpected indentation at line: {line}")

        if stripped.startswith("-"):
            if container is None:
                container = []
            elif not isinstance(container, list):
                raise ValueError("Cannot mix list and mapping entries")
            value_part = stripped[1:].strip()
            if value_part:
                items_list.append(_parse_scalar(value_part))
                index += 1
                continue
            item, index = _parse_block(lines, indent + 2, index + 1)
            items_list.append(item)
        else:
            if container is None:
                container = {}
            elif not isinstance(container, dict):
                raise ValueError("Cannot mix list and mapping entries")
            if ":" not in stripped:
                raise ValueError(f"Expected ':' in mapping line: {line}")
            key, value_part = _split_key_value(stripped)
            key = _parse_key(key)
            value_part = value_part.strip()
            if not key:
                raise ValueError("Empty key in mapping entry")
            if value_part:
                items_map[key] = _parse_scalar(value_part)
                index += 1
            else:
                value, index = _parse_block(lines, indent + 2, index + 1)
                items_map[key] = value
    if container is None:
        return {}, index
    if isinstance(container, list):
        return items_list, index
    return items_map, index


def _parse_scalar(token: str) -> Any:
    if token.startswith("\"") and token.endswith("\""):
        return token[1:-1]
    if token.startswith("'") and token.endswith("'"):
        return token[1:-1]
    lowered = token.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered == "null":
        return None
    # try integer
    try:
        return int(token)
    except ValueError:
        pass
    # try float
    try:
        return float(token)
    except ValueError:
        pass
    return token


def _parse_key(token: str) -> str:
    token = token.strip()
    if token.startswith("\"") and token.endswith("\""):
        return token[1:-1]
    if token.startswith("'") and token.endswith("'"):
        return token[1:-1]
    return token


def _split_key_value(line: str) -> tuple[str, str]:
    in_single = False
    in_double = False
    for index, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == ":" and not in_single and not in_double:
            return line[:index], line[index + 1 :]
    raise ValueError(f"Unable to split mapping entry: {line}")


class Config:
    """Convenience wrapper that exposes strongly-typed config sections."""

    def __init__(self, config_file: Any):
        self._path = Path(config_file)
        self._raw = load_config(self._path)
        self._agent = self._parse_agent(self._raw.get("agent", {}))
        self._tools = self._parse_tools(self._raw.get("tools", {}))

    @property
    def path(self) -> Path:
        return self._path

    @property
    def raw(self) -> Dict[str, Any]:
        return dict(self._raw)

    @property
    def agent(self) -> AgentSettings:
        return self._agent

    @property
    def tools(self) -> Dict[str, ToolSettings]:
        return dict(self._tools)

    def get_tool_settings(self, tool_name: str) -> ToolSettings:
        try:
            return self._tools[tool_name]
        except KeyError as exc:
            raise KeyError(f"Unknown tool '{tool_name}' referenced in plan") from exc

    def plan(self) -> List[Mapping[str, Any]]:
        return list(self._agent.plan)

    @staticmethod
    def _parse_agent(data: Mapping[str, Any]) -> AgentSettings:
        if not data:
            raise ValueError("Configuration is missing the 'agent' section")

        llm_data = data.get("llm") or {}
        if "provider" not in llm_data or "model" not in llm_data:
            raise ValueError("Agent configuration must include llm.provider and llm.model")

        llm = LLMConfig(
            provider=str(llm_data["provider"]),
            model=str(llm_data["model"]),
            parameters=dict(llm_data.get("parameters", {})),
        )

        tools = list(data.get("tools", []))
        plan = [dict(step) for step in data.get("plan", [])]
        name = str(data.get("name", "Agent"))
        return AgentSettings(name=name, llm=llm, tools=tools, plan=plan)

    @staticmethod
    def _parse_tools(data: Mapping[str, Any]) -> Dict[str, ToolSettings]:
        tools: Dict[str, ToolSettings] = {}
        for name, definition in data.items():
            if not isinstance(definition, Mapping):
                raise ValueError(f"Tool definition for '{name}' must be a mapping")
            class_path = definition.get("class")
            if not class_path:
                raise ValueError(f"Tool '{name}' is missing a 'class' entry")
            args = definition.get("args", {})
            if not isinstance(args, Mapping):
                raise ValueError(f"Tool '{name}' args must be a mapping")
            tools[name] = ToolSettings(class_path=str(class_path), args=dict(args))
        return tools
