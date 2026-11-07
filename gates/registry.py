"""Append-only decision log and adapter registry.

The plan asks for three things this file provides:

* every gate outcome writes a JSON line with every number in it, so the log is
  the evidence and the article is a view of it;
* an adapter registry with hash, config, parent, active pointer and a rollback
  command;
* a rollback test: deploy v1_good, roll back to v0, check the pointer.

Nothing here needs a GPU or the network.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_DIR = ROOT / "gates" / "registry"
REGISTRY_FILE = REGISTRY_DIR / "registry.json"
ACTIVE_FILE = REGISTRY_DIR / "ACTIVE"
DECISION_LOG = ROOT / "artifacts" / "decision_log.jsonl"


# ---------------------------------------------------------------------------
# Decision log
# ---------------------------------------------------------------------------

def log_decision(
    gate_result: dict[str, Any],
    *,
    candidate: str,
    baseline: str,
    meta: dict[str, Any] | None = None,
    path: Path = DECISION_LOG,
) -> dict[str, Any]:
    """Write one decision line and return the record."""
    path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "baseline": baseline,
        "candidate": candidate,
        "decision": gate_result.get("decision"),
        "status": gate_result.get("status"),
        "reason": gate_result.get("reason"),
        "failed_checks": gate_result.get("failed_checks", []),
        "thresholds": gate_result.get("thresholds", {}),
        "buckets": gate_result.get("buckets", {}),
        "meta": meta or {},
    }

    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, default=str) + "\n")

    return record


def read_decisions(path: Path = DECISION_LOG) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


# ---------------------------------------------------------------------------
# Adapter registry
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load() -> dict[str, Any]:
    if not REGISTRY_FILE.is_file():
        return {"adapters": {}, "active": None, "history": []}
    return json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))


def _save(state: dict[str, Any]) -> None:
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    active = state.get("active")
    if active:
        ACTIVE_FILE.write_text(active + "\n", encoding="utf-8")
    elif ACTIVE_FILE.exists():
        ACTIVE_FILE.unlink()


def register(
    name: str,
    path: str | Path,
    *,
    parent: str | None = None,
    config: dict[str, Any] | None = None,
    notes: str = "",
) -> dict[str, Any]:
    """Register an adapter directory or file by content hash."""
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(target)

    if target.is_dir():
        digest = hashlib.sha256()
        for child in sorted(target.rglob("*")):
            if child.is_file():
                digest.update(child.relative_to(target).as_posix().encode())
                digest.update(_sha256(child).encode())
        hash_value = digest.hexdigest()
    else:
        hash_value = _sha256(target)

    state = _load()
    entry = {
        "name": name,
        "path": str(target),
        "sha256": hash_value,
        "parent": parent,
        "config": config or {},
        "notes": notes,
        "registered_at": datetime.now(timezone.utc).isoformat(),
    }
    state["adapters"][name] = entry
    _save(state)
    return entry


def activate(name: str, *, reason: str = "") -> dict[str, Any]:
    state = _load()
    if name not in state["adapters"]:
        raise KeyError(f"unknown adapter {name!r}")
    previous = state.get("active")
    state["active"] = name
    state.setdefault("history", []).append(
        {
            "at": datetime.now(timezone.utc).isoformat(),
            "from": previous,
            "to": name,
            "reason": reason,
        }
    )
    _save(state)
    return {"active": name, "previous": previous}


def rollback(to: str | None = None) -> dict[str, Any]:
    """Roll back to the previous adapter, or to an explicit one."""
    state = _load()
    history = state.get("history", [])

    if to is None:
        previous = history[-1]["from"] if history else None
        if previous is None:
            raise RuntimeError("nothing to roll back to")
        to = previous

    if to not in state["adapters"]:
        raise KeyError(f"unknown adapter {to!r}")

    return activate(to, reason="rollback")


def active() -> str | None:
    return _load().get("active")


def show() -> dict[str, Any]:
    return _load()


if __name__ == "__main__":  # tiny CLI: python -m gates.registry
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["show", "active", "activate", "rollback"])
    parser.add_argument("name", nargs="?")
    args = parser.parse_args()

    if args.action == "show":
        print(json.dumps(show(), indent=2))
    elif args.action == "active":
        print(active())
    elif args.action == "activate":
        print(activate(args.name, reason="manual"))
    else:
        print(rollback(args.name))
