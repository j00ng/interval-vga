"""DXF layer semantics for height-aware visibility.

Geometry extraction and architectural meaning are deliberately separate. Layer
names provide defaults; an optional JSON sidecar can override kind/sill/head per
layer without putting school-specific roles into the visibility core.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import re
from pathlib import Path


class LayerType(str, Enum):
    WALL = "wall"
    DOOR = "door"
    WINDOW = "window"
    UNKNOWN_OPENING = "unknown_opening"


@dataclass(frozen=True)
class OpeningProfile:
    """Vertical intervals are configuration, not visibility-core constants."""

    door: tuple[float, float] = (0.0, 2.1)
    window: tuple[float, float] = (1.2, 2.1)

    def __post_init__(self) -> None:
        for name, (low, high) in (("door", self.door), ("window", self.window)):
            if low > high:
                raise ValueError(f"{name} interval must satisfy z_min <= z_max")

    def interval_for(self, kind: LayerType) -> tuple[float, float]:
        if kind is LayerType.DOOR:
            return self.door
        if kind is LayerType.WINDOW:
            return self.window
        raise ValueError(f"no vertical interval for {kind.value}")


DEFAULT_LAYER_ALIASES = {
    "WALL": LayerType.WALL,
    "DOOR": LayerType.DOOR,
    "WINDOW": LayerType.WINDOW,
}


@dataclass(frozen=True)
class LayerSemantic:
    kind: LayerType
    sill: float | None = None
    head: float | None = None
    source: str = "layer-name"


@dataclass(frozen=True)
class ParsedDXF:
    walls: list[tuple[float, float, float, float]]
    openings: list[tuple[float, float, float, float, float, float]]
    unknown_openings: list[tuple[str, tuple[float, float, float, float]]]
    layer_semantics: dict[str, LayerSemantic]

    @property
    def unknown_opening_layers(self) -> tuple[str, ...]:
        return tuple(sorted({layer for layer, _ in self.unknown_openings}))


@dataclass(frozen=True)
class SemanticConfig:
    """Drawing-specific names and heights, kept outside the LOS engine."""

    profile: OpeningProfile = OpeningProfile()
    aliases: dict[str, LayerType] | None = None
    layers: dict[str, LayerSemantic] | None = None


_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"


def _height_from_name(name: str, key: str) -> float | None:
    match = re.search(rf"(?:^|[_\- ]){key}(?:=|:|[_\- ])?({_NUMBER})(?:$|[_\- ])", name, re.I)
    return float(match.group(1)) if match else None


def infer_layer_semantic(
    layer: str,
    *,
    profile: OpeningProfile = OpeningProfile(),
    aliases: dict[str, LayerType | str] | None = None,
) -> LayerSemantic:
    upper = layer.upper()
    alias_map = dict(DEFAULT_LAYER_ALIASES)
    for alias, kind in (aliases or {}).items():
        alias_map[alias.upper()] = kind if isinstance(kind, LayerType) else LayerType(kind)
    sill = _height_from_name(layer, "SILL")
    head = _height_from_name(layer, "HEAD")
    kind = alias_map.get(upper)
    if kind is None:
        tokens = {token for token in re.split(r"[^A-Z0-9]+", upper) if token}
        if "WINDOW" in tokens or "WIN" in tokens:
            kind = LayerType.WINDOW
        elif "DOOR" in tokens:
            kind = LayerType.DOOR
        elif "WALL" in tokens:
            kind = LayerType.WALL
    if kind in (LayerType.DOOR, LayerType.WINDOW):
        low, high = profile.interval_for(kind)
        return LayerSemantic(kind, low if sill is None else sill, high if head is None else head)
    if kind is LayerType.WALL:
        return LayerSemantic(LayerType.WALL)
    if "OPEN" in upper:
        return LayerSemantic(LayerType.UNKNOWN_OPENING, source="unresolved")
    return LayerSemantic(LayerType.WALL)


def _profile_interval(value, *, name: str) -> tuple[float, float]:
    if isinstance(value, dict):
        low, high = value.get("sill"), value.get("head")
    else:
        try:
            low, high = value
        except (TypeError, ValueError) as error:
            raise ValueError(f"opening_profile.{name} must contain sill/head") from error
    if low is None or high is None or float(low) > float(high):
        raise ValueError(f"opening_profile.{name} requires sill <= head")
    return float(low), float(high)


def load_semantic_config(
    path: str | None, *, base_profile: OpeningProfile = OpeningProfile()
) -> SemanticConfig:
    if not path:
        return SemanticConfig(profile=base_profile, aliases={}, layers={})
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    profile_raw = raw.get("opening_profile", {})
    profile = OpeningProfile(
        door=_profile_interval(profile_raw["door"], name="door")
        if "door" in profile_raw else base_profile.door,
        window=_profile_interval(profile_raw["window"], name="window")
        if "window" in profile_raw else base_profile.window,
    )
    aliases = {
        name.upper(): LayerType(str(kind).lower())
        for name, kind in raw.get("aliases", {}).items()
    }
    # The old {"LAYER": {...}} form remains accepted for existing projects.
    layers = raw.get("layers", raw if not any(
        key in raw for key in ("opening_profile", "aliases")
    ) else {})
    result: dict[str, LayerSemantic] = {}
    for name, value in layers.items():
        kind = LayerType(str(value["kind"]).lower())
        sill = value.get("sill")
        head = value.get("head")
        if kind in (LayerType.DOOR, LayerType.WINDOW):
            default_low, default_high = profile.interval_for(kind)
            sill = default_low if sill is None else sill
            head = default_high if head is None else head
        if sill is not None and head is not None and float(sill) > float(head):
            raise ValueError(f"layer {name} requires sill <= head")
        result[name.upper()] = LayerSemantic(
            kind,
            None if sill is None else float(sill),
            None if head is None else float(head),
            source="metadata",
        )
    return SemanticConfig(profile=profile, aliases=aliases, layers=result)


def load_layer_metadata(path: str | None) -> dict[str, LayerSemantic]:
    """Backward-compatible layer-only view of a semantic sidecar."""
    return load_semantic_config(path).layers or {}


def _line_entities(path: str):
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    tokens = [token.strip() for token in text.splitlines()]
    index = 0
    while index < len(tokens) - 1:
        if tokens[index] == "0" and tokens[index + 1] == "LINE":
            layer, values = "0", {}
            cursor = index + 2
            while cursor < len(tokens) - 1 and tokens[cursor] != "0":
                code, value = tokens[cursor], tokens[cursor + 1]
                if code == "8":
                    layer = value
                elif code in ("10", "20", "11", "21"):
                    try:
                        values[code] = float(value)
                    except ValueError:
                        pass
                cursor += 2
            if len(values) == 4:
                yield layer, (values["10"], values["20"], values["11"], values["21"])
            index = cursor
        else:
            index += 1


def parse_dxf(
    path: str,
    *,
    metadata_path: str | None = None,
    profile: OpeningProfile = OpeningProfile(),
    aliases: dict[str, LayerType | str] | None = None,
) -> ParsedDXF:
    config = load_semantic_config(metadata_path, base_profile=profile)
    overrides = config.layers or {}
    effective_aliases = dict(config.aliases or {})
    effective_aliases.update(aliases or {})
    walls = []
    openings = []
    unknown_openings = []
    semantics: dict[str, LayerSemantic] = {}
    for layer, segment in _line_entities(path):
        key = layer.upper()
        semantic = overrides.get(key) or infer_layer_semantic(
            layer, profile=config.profile, aliases=effective_aliases
        )
        semantics[layer] = semantic
        if semantic.kind is LayerType.WALL:
            walls.append(segment)
        elif semantic.kind is LayerType.UNKNOWN_OPENING:
            unknown_openings.append((layer, segment))
        else:
            openings.append((*segment, float(semantic.sill), float(semantic.head)))
    return ParsedDXF(walls, openings, unknown_openings, semantics)
