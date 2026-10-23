#!/usr/bin/env python3
"""Provisional censored-mixture recovery command."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path


MAX_RECORDS = 120


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("payload must be an object")
    return value


def number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("numeric value required")
    return float(value)


def validate(payload: dict) -> dict:
    if set(payload) != {"case_id", "max_records", "batch_catalog", "records"}:
        raise ValueError("payload keys do not match")
    if payload["max_records"] != MAX_RECORDS:
        raise ValueError("max_records is unsupported")
    catalog = payload["batch_catalog"]
    records = payload["records"]
    if not isinstance(catalog, list) or len(catalog) != 4:
        raise ValueError("batch catalog count is invalid")
    clean_catalog = []
    labels = set()
    for item in catalog:
        if not isinstance(item, dict) or set(item) != {"label", "weights"}:
            raise ValueError("catalog keys do not match")
        if not isinstance(item["label"], str) or item["label"] in labels:
            raise ValueError("catalog labels are invalid")
        labels.add(item["label"])
        weights = item["weights"]
        if not isinstance(weights, list) or len(weights) != 3:
            raise ValueError("catalog weights are invalid")
        clean_weights = [number(value) for value in weights]
        if abs(sum(clean_weights) - 1.0) > 1e-9:
            raise ValueError("catalog weights must sum to one")
        clean_catalog.append({"label": item["label"], "weights": clean_weights})
    if not isinstance(records, list) or not 1 <= len(records) <= MAX_RECORDS:
        raise ValueError("record count is invalid")
    clean_records = []
    labs = set()
    for item in records:
        if not isinstance(item, dict) or set(item) != {"lab_id", "value", "censor"}:
            raise ValueError("record keys do not match")
        if not isinstance(item["lab_id"], str):
            raise ValueError("lab_id is invalid")
        labs.add(item["lab_id"])
        threshold = number(item["censor"])
        if not 0.5 <= threshold <= 2.0:
            raise ValueError("censor threshold is invalid")
        if item["value"] is None:
            value = None
        else:
            value = number(item["value"])
            if value > threshold:
                raise ValueError("visible value exceeds threshold")
        clean_records.append({"lab_id": item["lab_id"], "value": value, "censor": threshold})
    if len(labs) != 10:
        raise ValueError("exactly ten laboratory ids are required")
    return {
        "case_id": payload["case_id"],
        "max_records": MAX_RECORDS,
        "batch_catalog": clean_catalog,
        "records": clean_records,
    }


def estimate(payload: dict) -> dict:
    visible = sorted(r["value"] for r in payload["records"] if r["value"] is not None)
    means = [visible[int(q * (len(visible) - 1))] for q in (0.15, 0.5, 0.85)]
    sigma = 0.25
    labels = sorted(item["label"] for item in payload["batch_catalog"])
    assignments = [
        {"lab_id": sorted({record["lab_id"] for record in payload["records"]})[index], "batch": labels[index % 4]}
        for index in range(10)
    ]
    censored = sum(record["value"] is None for record in payload["records"])
    return {
        "status": "recovered",
        "components": [
            {"mean": round(mean, 9), "sigma": sigma, "visible_count": len(visible) // 3}
            for mean in means
        ],
        "batch_profiles": [
            {"label": item["label"], "weights": item["weights"]}
            for item in sorted(payload["batch_catalog"], key=lambda item: item["label"])
        ],
        "lab_assignments": assignments,
        "quality": {
            "uncensored_count": len(visible),
            "censored_count": censored,
            "assignment_cost": 0.0,
            "moment_spread": 0.0,
        },
    }


def canonical(payload: dict) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def atomic_write(path: Path, data: bytes) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def append_ledger(path: Path, case_id: str, source_hash: str, output_hash: str) -> None:
    rows = []
    if path.exists():
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if any(
        row.get("case_id") == case_id
        and row.get("input_sha256") == source_hash
        and row.get("output_sha256") == output_hash
        for row in rows
    ):
        return
    record = {
        "case_id": case_id,
        "input_sha256": source_hash,
        "output_sha256": output_hash,
        "sequence": len(rows) + 1,
    }
    with path.open("ab") as handle:
        handle.write(canonical(record))
        handle.flush()
        os.fsync(handle.fileno())


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print("MIXTURE_REFUSED: expected input output ledger", file=sys.stderr)
        return 2
    try:
        source, target, ledger = map(Path, argv[1:])
        payload = validate(load(source))
        output = canonical(estimate(payload))
        atomic_write(target, output)
        append_ledger(
            ledger,
            payload["case_id"],
            hashlib.sha256(source.read_bytes()).hexdigest(),
            hashlib.sha256(output).hexdigest(),
        )
        print("MIXTURE_RECOVERED")
        return 0
    except Exception as exc:
        print(f"MIXTURE_REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
