"""Shared utilities for petro_mcp."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


class PathNotAllowedError(Exception):
    """Raised when a target path falls outside every allowlisted root."""


def _resolve(p: Path) -> Path:
    return Path(p).expanduser().resolve()


def validate_path(
    target: Path | str,
    allowed: Sequence[Path | str] | None = None,
) -> Path:
    """Return the resolved target if it lives inside any allowed root.

    Path validation is opt-in. When ``allowed`` is None or empty, the target
    is resolved and returned unchanged (permissive mode). When ``allowed`` is
    provided, the target's resolved path must live inside at least one of the
    allowed roots, otherwise ``PathNotAllowedError`` is raised.

    The target is resolved (symlinks followed) before the containment check so
    that symlinks cannot escape an allowlisted directory.

    Args:
        target: File or directory path to validate.
        allowed: Optional iterable of allowed root directories.

    Raises:
        FileNotFoundError: if ``target`` does not exist on disk.
        PathNotAllowedError: if ``allowed`` is non-empty and the resolved
            target is not inside any allowed root.
    """
    target_path = _resolve(Path(target))
    if not target_path.exists():
        raise FileNotFoundError(str(target_path))

    if not allowed:
        return target_path

    allowed_resolved = [_resolve(Path(a)) for a in allowed]
    for root in allowed_resolved:
        try:
            target_path.relative_to(root)
            return target_path
        except ValueError:
            continue

    raise PathNotAllowedError(
        f"path {target_path} is not inside any allowed root "
        f"(symlinks are resolved before this check, so the displayed path "
        f"may differ from the literal input)"
    )
