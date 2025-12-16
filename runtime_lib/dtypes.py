"""Shared helpers for normalizing dtype options across projects."""

from __future__ import annotations

from typing import Any, NamedTuple

import torch

_TORCH_IMPORT_ERROR: Exception | None = None


class DtypeSpec(NamedTuple):
    """Normalized representation of dtype fields.

    Attributes
    ----------
    label:
        Canonical lowercase label (e.g., ``"float32"``) or ``"auto"``/``None``.
    torch:
        The resolved ``torch.dtype`` object.
    """

    label: str | None
    torch: Any


_CANONICAL_FROM_ALIAS: dict[str, str] = {
    "float16": "float16",
    "half": "float16",
    "fp16": "float16",
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
    "float32": "float32",
    "float": "float32",
    "single": "float32",
    "fp32": "float32",
    "float64": "float64",
    "double": "float64",
    "fp64": "float64",
}

_TORCH_FROM_CANONICAL: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}


def coerce_dtype_spec(value: Any) -> DtypeSpec | None:
    """Convert a config field into a :class:`DtypeSpec`.

    Returns ``None`` when the input is ``None``/empty.
    """

    if value is None:
        return None
    if isinstance(value, DtypeSpec):
        return value

    if isinstance(value, torch.dtype):
        return DtypeSpec(str(value).split(".")[-1], value)

    if isinstance(value, str):
        trimmed = value.strip()
        if not trimmed:
            return None
        lowered = trimmed.lower()
        if lowered == "auto":
            return DtypeSpec("auto", None)
        canonical = _CANONICAL_FROM_ALIAS.get(lowered)
        if canonical is None:
            allowed = sorted(set(_CANONICAL_FROM_ALIAS)) + ["auto"]
            raise ValueError(
                f"Unsupported dtype '{value}'. Allowed values: {', '.join(allowed)}"
            )
        return DtypeSpec(canonical, _TORCH_FROM_CANONICAL[canonical])

    if isinstance(value, dict):
        # Accept user-provided structures like {"value": "float32"}.
        for key in ("value", "label", "raw"):
            if key in value:
                return coerce_dtype_spec(value[key])
        if "torch" in value:
            return DtypeSpec(value.get("label"), value["torch"])

    raise TypeError(f"Cannot coerce dtype value of type {type(value)!r}")


def resolve_config_dtypes(config: dict[str, Any]) -> dict[str, Any]:
    """Walk the config dict and normalize any ``dtype`` fields in-place."""

    def _walk(node: Any) -> Any:
        if isinstance(node, dict):
            for key, value in list(node.items()):
                if key == "dtype":
                    node[key] = coerce_dtype_spec(value)
                else:
                    node[key] = _walk(value)
        elif isinstance(node, list):
            for idx, entry in enumerate(node):
                node[idx] = _walk(entry)
        return node

    return _walk(config)


def dtype_label(value: Any) -> str | None:
    """Return the canonical label (if any) for a dtype field."""

    if isinstance(value, DtypeSpec):
        return value.label
    if isinstance(value, str):
        lowered = value.strip().lower()
        return lowered or None
    return None


def dtype_torch(value: Any) -> Any:
    """Return the ``torch.dtype`` associated with the field (if resolved)."""

    if isinstance(value, DtypeSpec):
        return value.torch
    if isinstance(value, torch.dtype):
        return value
    return None


def describe_dtype(value: Any) -> str:
    """Human readable description used for logging."""

    label = dtype_label(value) or "unspecified"
    torch_dtype = dtype_torch(value)
    if torch_dtype is None:
        return label
    return f"{label} ({torch_dtype})"


__all__ = [
    "DtypeSpec",
    "coerce_dtype_spec",
    "resolve_config_dtypes",
    "dtype_label",
    "dtype_torch",
    "describe_dtype",
]
