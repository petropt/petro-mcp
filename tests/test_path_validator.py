"""Tests for the optional path-allowlist validator."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from petro_mcp.utils import PathNotAllowedError, validate_path


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "wells").mkdir()
    (tmp_path / "wells" / "a.las").write_text("dummy")
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "b.las").write_text("dummy")
    return tmp_path


def test_permissive_when_allowed_is_none(workspace: Path):
    target = workspace / "wells" / "a.las"
    result = validate_path(target, None)
    assert result == target.resolve()


def test_permissive_when_allowed_is_empty_list(workspace: Path):
    target = workspace / "wells" / "a.las"
    result = validate_path(target, [])
    assert result == target.resolve()


def test_allow_inside_root(workspace: Path):
    target = workspace / "wells" / "a.las"
    result = validate_path(target, [workspace / "wells"])
    assert result == target.resolve()


def test_deny_outside_root(workspace: Path):
    target = workspace / "other" / "b.las"
    with pytest.raises(PathNotAllowedError):
        validate_path(target, [workspace / "wells"])


def test_multiple_roots_allows_first_match(workspace: Path):
    target = workspace / "other" / "b.las"
    result = validate_path(target, [workspace / "wells", workspace / "other"])
    assert result == target.resolve()


def test_nonexistent_target_raises(workspace: Path):
    target = workspace / "wells" / "missing.las"
    with pytest.raises(FileNotFoundError):
        validate_path(target, [workspace / "wells"])


def test_symlink_resolved_before_check(workspace: Path):
    # A symlink inside an allowed dir that points outside that dir
    # should be rejected, because the target is resolved first.
    sneaky = workspace / "wells" / "sneaky.las"
    real_outside = workspace / "other" / "b.las"
    try:
        os.symlink(real_outside, sneaky)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported in this environment")
    with pytest.raises(PathNotAllowedError):
        validate_path(sneaky, [workspace / "wells"])


def test_accepts_str_and_path(workspace: Path):
    target = workspace / "wells" / "a.las"
    assert validate_path(str(target), [str(workspace / "wells")]) == target.resolve()
    assert validate_path(target, [workspace / "wells"]) == target.resolve()


def test_expands_tilde():
    # Resolves ~ without raising; non-existent home subpath should FileNotFoundError
    with pytest.raises(FileNotFoundError):
        validate_path("~/__definitely_not_a_real_path_12345__", None)
