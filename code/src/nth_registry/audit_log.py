"""Canonical append-only hash-chain records for off-chain decisions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


GENESIS_HASH = "0" * 64


def canonical_json(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


@dataclass(frozen=True)
class DecisionRecord:
    request_id: str
    probe_sha256: str
    candidate_commitments: tuple[str, ...]
    nth_distances: tuple[int, ...]
    embedding_distances: tuple[float, ...]
    matched_commitment: str | None
    nth_version: str
    nth_radius: int
    embedding_model_sha256: str
    embedding_threshold: float
    decision: str
    policy_sha256: str
    previous_record_hash: str = GENESIS_HASH
    timestamp_utc: str = ""

    def payload(self) -> dict[str, Any]:
        value = asdict(self)
        if not value["timestamp_utc"]:
            value["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        return value


def record_hash(previous_hash: str, body: dict[str, Any]) -> str:
    """Hash ``previous_head_bytes || canonical_json(record_body)``."""

    if len(previous_hash) != 64:
        raise ValueError("previous_hash must be a 64-character SHA-256 hex digest")
    try:
        previous_bytes = bytes.fromhex(previous_hash)
    except ValueError as error:
        raise ValueError("previous_hash must be hexadecimal") from error
    return hashlib.sha256(previous_bytes + canonical_json(body)).hexdigest()


def record_body(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the canonical body; link and derived hash are stored separately."""

    return {
        key: value
        for key, value in payload.items()
        if key not in {"previous_record_hash", "record_hash"}
    }


def append_record(path: str | Path, record: DecisionRecord) -> str:
    destination = Path(path)
    previous = GENESIS_HASH
    if destination.exists() and destination.stat().st_size:
        last = destination.read_text(encoding="utf-8").strip().splitlines()[-1]
        previous = json.loads(last)["record_hash"]
    payload = record.payload()
    payload["previous_record_hash"] = previous
    payload["record_hash"] = record_hash(previous, record_body(payload))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
    return payload["record_hash"]


def verify_chain(path: str | Path) -> tuple[bool, int, str]:
    previous = GENESIS_HASH
    count = 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        stored = row.pop("record_hash")
        linked = row.pop("previous_record_hash", None)
        if linked != previous or record_hash(previous, record_body(row)) != stored:
            return False, count, previous
        previous = stored
        count += 1
    return True, count, previous
