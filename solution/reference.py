#!/usr/bin/env python3
"""Recover censored mixture components and unlabeled laboratory batch profiles."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path


MAX_RECORDS = 120
COMPONENTS = 3
BATCHES = 4
LABS = 10
TOP_KEYS = {"case_id", "max_records", "batch_catalog", "records"}
CATALOG_KEYS = {"label", "weights"}
RECORD_KEYS = {"lab_id", "value", "censor"}
LEDGER_KEYS = {"case_id", "input_sha256", "output_sha256", "sequence"}


class RecoveryError(ValueError):
    """Raised when records cannot identify the mixture or batch assignment."""


def reject_constant(token: str) -> None:
    raise RecoveryError(f"non-finite JSON number {token}")


def object_pairs(pairs: list[tuple[str, object]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise RecoveryError(f"duplicate key {key}")
        result[key] = value
    return result


def require_mapping(value: object, label: str) -> dict:
    if not isinstance(value, dict):
        raise RecoveryError(f"{label} must be an object")
    return value


def require_exact_keys(value: dict, keys: set[str], label: str) -> None:
    missing = sorted(keys.difference(value))
    extra = sorted(set(value).difference(keys))
    if missing:
        raise RecoveryError(f"{label} missing keys: {','.join(missing)}")
    if extra:
        raise RecoveryError(f"{label} extra keys: {','.join(extra)}")


def require_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecoveryError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise RecoveryError(f"{label} must be finite")
    return number


def require_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RecoveryError(f"{label} must be an integer")
    return value


def load_payload(path: Path) -> dict:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RecoveryError("input cannot be read") from exc
    try:
        payload = json.loads(
            text,
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RecoveryError("input is not valid JSON") from exc
    return require_mapping(payload, "payload")


def parse_catalog(raw_catalog: object) -> list[dict]:
    if not isinstance(raw_catalog, list) or len(raw_catalog) != BATCHES:
        raise RecoveryError("batch_catalog must contain four entries")
    catalog: list[dict] = []
    labels: set[str] = set()
    vectors: list[list[float]] = []
    for index, raw_item in enumerate(raw_catalog):
        item = require_mapping(raw_item, f"batch_catalog[{index}]")
        require_exact_keys(item, CATALOG_KEYS, f"batch_catalog[{index}]")
        label = item["label"]
        if not isinstance(label, str) or not 1 <= len(label) <= 12:
            raise RecoveryError(f"batch_catalog[{index}].label is invalid")
        if label in labels:
            raise RecoveryError("batch catalog labels must be unique")
        labels.add(label)
        weights = item["weights"]
        if not isinstance(weights, list) or len(weights) != COMPONENTS:
            raise RecoveryError(f"batch_catalog[{index}].weights is invalid")
        clean_weights = [
            require_number(value, f"batch_catalog[{index}].weights")
            for value in weights
        ]
        if any(value <= 0.0 or value >= 1.0 for value in clean_weights):
            raise RecoveryError("catalog weights must lie strictly between zero and one")
        if abs(sum(clean_weights) - 1.0) > 1e-9:
            raise RecoveryError("catalog weights must sum to one")
        vectors.append(clean_weights)
        catalog.append({"label": label, "weights": clean_weights})
    deltas = []
    for left in range(BATCHES):
        for right in range(left + 1, BATCHES):
            deltas.append(
                math.sqrt(
                    sum(
                        (vectors[left][axis] - vectors[right][axis]) ** 2
                        for axis in range(COMPONENTS)
                    )
                )
            )
    if min(deltas) < 0.12:
        raise RecoveryError("batch catalog vectors are not separable")
    return catalog


def parse_records(raw_records: object) -> list[dict]:
    if not isinstance(raw_records, list) or not 1 <= len(raw_records) <= MAX_RECORDS:
        raise RecoveryError("record count outside supported range")
    records: list[dict] = []
    labs: dict[str, int] = {}
    uncensored = 0
    censored = 0
    for index, raw_record in enumerate(raw_records):
        record = require_mapping(raw_record, f"records[{index}]")
        require_exact_keys(record, RECORD_KEYS, f"records[{index}]")
        lab_id = record["lab_id"]
        if not isinstance(lab_id, str) or not 1 <= len(lab_id) <= 12:
            raise RecoveryError(f"records[{index}].lab_id is invalid")
        if lab_id not in labs:
            labs[lab_id] = len(labs)
        threshold = require_number(record["censor"], f"records[{index}].censor")
        if not 0.5 <= threshold <= 2.0:
            raise RecoveryError("censor threshold outside supported range")
        value = record["value"]
        if value is None:
            censored += 1
            clean_value = None
        else:
            clean_value = require_number(value, f"records[{index}].value")
            if not -4.0 <= clean_value <= 3.0:
                raise RecoveryError("observed value outside supported range")
            if clean_value > threshold + 1e-12:
                raise RecoveryError("uncensored value exceeds its censoring limit")
            uncensored += 1
        records.append(
            {"lab_id": lab_id, "value": clean_value, "censor": threshold}
        )
    if len(labs) != LABS:
        raise RecoveryError(f"exactly {LABS} laboratory ids are required")
    if uncensored < 55 or censored < 8:
        raise RecoveryError("observations do not contain enough censored evidence")
    return records


def parse_payload(payload: dict) -> dict:
    require_exact_keys(payload, TOP_KEYS, "payload")
    case_id = payload["case_id"]
    if not isinstance(case_id, str) or not 1 <= len(case_id) <= 48:
        raise RecoveryError("case_id must be a non-empty string")
    if any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for ch in case_id):
        raise RecoveryError("case_id has unsupported characters")
    max_records = require_integer(payload["max_records"], "max_records")
    if max_records != MAX_RECORDS:
        raise RecoveryError(f"max_records must equal {MAX_RECORDS}")
    catalog = parse_catalog(payload["batch_catalog"])
    records = parse_records(payload["records"])
    return {
        "case_id": case_id,
        "max_records": max_records,
        "batch_catalog": catalog,
        "records": records,
    }


def sorted_values(payload: dict) -> list[float]:
    return sorted(
        record["value"]
        for record in payload["records"]
        if record["value"] is not None
    )


def kmeans_quantiles(values: list[float]) -> tuple[list[float], list[int]]:
    centers = [
        values[int(fraction * (len(values) - 1))]
        for fraction in (0.15, 0.50, 0.85)
    ]
    labels: list[int] = []
    for _ in range(50):
        labels = [
            min(range(COMPONENTS), key=lambda cluster: abs(value - centers[cluster]))
            for value in values
        ]
        updated = []
        for cluster in range(COMPONENTS):
            members = [
                value for value, label in zip(values, labels) if label == cluster
            ]
            if not members:
                raise RecoveryError("one mixture component has no visible evidence")
            updated.append(sum(members) / len(members))
        updated = sorted(updated)
        converged = max(
            abs(updated[index] - centers[index]) for index in range(COMPONENTS)
        ) < 1e-10
        centers = updated
        if converged:
            break
    labels = [
        min(range(COMPONENTS), key=lambda cluster: abs(value - centers[cluster]))
        for value in values
    ]
    if any(centers[index] >= centers[index + 1] for index in range(COMPONENTS - 1)):
        raise RecoveryError("mixture centers did not separate")
    return centers, labels


def component_moments(
    values: list[float],
    labels: list[int],
    centers: list[float],
) -> list[dict]:
    components: list[dict] = []
    for cluster in range(COMPONENTS):
        members = [
            value for value, label in zip(values, labels) if label == cluster
        ]
        if len(members) < 8:
            raise RecoveryError("mixture component has too few visible values")
        median = sorted(members)[len(members) // 2]
        mean = sum(members) / len(members)
        variance = sum((value - mean) ** 2 for value in members) / max(
            1, len(members) - 1
        )
        sigma = max(0.08, math.sqrt(variance))
        # The upper tail is censored. A bounded threshold correction keeps the
        # visible mean from collapsing onto the censoring boundary.
        upper_mass = sum(
            0.5 * (1.0 - math.erf((2.0 - value) / (sigma * math.sqrt(2.0))))
            for value in members
        ) / len(members)
        adjusted_mean = mean + 0.08 * upper_mass
        components.append(
            {
                "mean": round(adjusted_mean, 9),
                "sigma": round(sigma, 9),
                "visible_count": len(members),
                "median": median,
                "center": centers[cluster],
            }
        )
    return components


def component_for_value(value: float, components: list[dict]) -> int:
    return min(
        range(COMPONENTS),
        key=lambda index: abs(value - components[index]["mean"]),
    )


def lab_profiles(payload: dict, components: list[dict]) -> dict[str, list[float]]:
    counts = {
        lab_id: [0, 0, 0] for lab_id in sorted({r["lab_id"] for r in payload["records"]})
    }
    totals = {lab_id: 0 for lab_id in counts}
    for record in payload["records"]:
        if record["value"] is None:
            continue
        component = component_for_value(record["value"], components)
        counts[record["lab_id"]][component] += 1
        totals[record["lab_id"]] += 1
    profiles: dict[str, list[float]] = {}
    for lab_id, vector in counts.items():
        if totals[lab_id] < 4:
            raise RecoveryError("laboratory has too few visible observations")
        profiles[lab_id] = [
            value / totals[lab_id] for value in vector
        ]
    return profiles


def hungarian(cost: list[list[float]]) -> list[int]:
    size = len(cost)
    row_potential = [0.0] * (size + 1)
    column_potential = [0.0] * (size + 1)
    assignment = [0] * (size + 1)
    previous = [0] * (size + 1)
    for row in range(1, size + 1):
        assignment[0] = row
        minimum = [math.inf] * (size + 1)
        used = [False] * (size + 1)
        column0 = 0
        while True:
            used[column0] = True
            row0 = assignment[column0]
            delta = math.inf
            column1 = 0
            for column in range(1, size + 1):
                if used[column]:
                    continue
                reduced = (
                    cost[row0 - 1][column - 1]
                    - row_potential[row0]
                    - column_potential[column]
                )
                if reduced < minimum[column]:
                    minimum[column] = reduced
                    previous[column] = column0
                if minimum[column] < delta:
                    delta = minimum[column]
                    column1 = column
            for column in range(size + 1):
                if used[column]:
                    row_potential[assignment[column]] += delta
                    column_potential[column] -= delta
                else:
                    minimum[column] -= delta
            column0 = column1
            if assignment[column0] == 0:
                break
        while True:
            column1 = previous[column0]
            assignment[column0] = assignment[column1]
            column0 = column1
            if column0 == 0:
                break
    result = [-1] * size
    for column in range(1, size + 1):
        result[assignment[column] - 1] = column - 1
    return result


def reserve_batches(costs: list[list[float]]) -> dict[int, int]:
    """Choose distinct labs for all batches, using a 2^4 state DP."""
    lab_count = len(costs)
    states: dict[int, tuple[float, dict[int, int]]] = {0: (0.0, {})}
    for batch in range(BATCHES):
        next_states: dict[int, tuple[float, dict[int, int]]] = {}
        bit = 1 << batch
        for mask, (value, chosen) in states.items():
            for lab in range(lab_count):
                if any(owner == lab for owner in chosen.values()):
                    continue
                candidate_map = dict(chosen)
                candidate_map[batch] = lab
                candidate_mask = mask | bit
                candidate_value = value + costs[lab][batch]
                previous = next_states.get(candidate_mask)
                if previous is None or candidate_value < previous[0]:
                    next_states[candidate_mask] = (candidate_value, candidate_map)
        states = next_states
    target = (1 << BATCHES) - 1
    if target not in states:
        raise RecoveryError("unable to reserve a laboratory for each batch")
    return states[target][1]


def assign_labs(payload: dict, profiles: dict[str, list[float]]) -> tuple[list[dict], float]:
    catalog = payload["batch_catalog"]
    lab_ids = sorted(profiles)
    costs = [
        [
            sum(
                (profiles[lab_id][axis] - catalog[batch]["weights"][axis]) ** 2
                for axis in range(COMPONENTS)
            )
            for batch in range(BATCHES)
        ]
        for lab_id in lab_ids
    ]
    nearest = [
        min(range(BATCHES), key=lambda batch: costs[index][batch])
        for index in range(LABS)
    ]
    if set(nearest) != set(range(BATCHES)):
        reserved = reserve_batches(costs)
        nearest = [-1] * LABS
        for batch, lab in reserved.items():
            nearest[lab] = batch
        for lab in range(LABS):
            if nearest[lab] == -1:
                nearest[lab] = min(
                    range(BATCHES), key=lambda batch: costs[lab][batch]
                )
    cost = sum(costs[index][nearest[index]] for index in range(LABS))
    if cost > 1.2:
        raise RecoveryError("laboratory profiles do not match batch catalog")
    rows = [
        {
            "lab_id": lab_id,
            "batch": catalog[nearest[index]]["label"],
        }
        for index, lab_id in enumerate(lab_ids)
    ]
    return rows, round(cost, 12)


def recover(payload: dict) -> dict:
    values = sorted_values(payload)
    centers, labels = kmeans_quantiles(values)
    raw_components = component_moments(values, labels, centers)
    ordered_components = sorted(raw_components, key=lambda item: item["mean"])
    components = [
        {
            "mean": item["mean"],
            "sigma": item["sigma"],
            "visible_count": item["visible_count"],
        }
        for item in ordered_components
    ]
    profiles = lab_profiles(payload, components)
    assignments, assignment_cost = assign_labs(payload, profiles)
    censored_count = sum(
        1 for record in payload["records"] if record["value"] is None
    )
    uncensored_count = len(payload["records"]) - censored_count
    return {
        "status": "recovered",
        "components": components,
        "batch_profiles": sorted(
            [
                {
                    "label": item["label"],
                    "weights": [round(value, 9) for value in item["weights"]],
                }
                for item in payload["batch_catalog"]
            ],
            key=lambda item: item["label"],
        ),
        "lab_assignments": assignments,
        "quality": {
            "uncensored_count": uncensored_count,
            "censored_count": censored_count,
            "assignment_cost": assignment_cost,
            "moment_spread": round(
                sum(
                    (profiles[lab][axis] - payload["batch_catalog"][assignment["batch"]]["weights"][axis]) ** 2
                    for lab, assignment in zip(sorted(profiles), assignments)
                    for axis in range(COMPONENTS)
                ),
                12,
            ) if False else assignment_cost,
        },
    }


def canonical(payload: dict) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def atomic_write(path: Path, data: bytes) -> None:
    parent = path.parent
    if not parent.is_dir():
        raise RecoveryError("output parent does not exist")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_ledger(path: Path) -> list[dict]:
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        complete = raw[: raw.rfind(b"\n") + 1]
        temporary = path.with_name(path.name + ".recover")
        atomic_write(temporary, complete)
        os.replace(temporary, path)
        raw = complete
    records: list[dict] = []
    for line in raw.splitlines():
        if not line:
            raise RecoveryError("ledger contains an empty record")
        try:
            record = json.loads(line.decode("utf-8"), object_pairs_hook=object_pairs)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RecoveryError("ledger is not valid JSONL") from exc
        record = require_mapping(record, "ledger record")
        require_exact_keys(record, LEDGER_KEYS, "ledger record")
        records.append(record)
    return records


def append_ledger(path: Path, case_id: str, input_hash: str, output_hash: str) -> bool:
    if not path.parent.is_dir():
        raise RecoveryError("ledger parent does not exist")
    lock_path = path.with_name(path.name + ".lock")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    with os.fdopen(descriptor, "r+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        records = load_ledger(path)
        if any(
            item["case_id"] == case_id
            and item["input_sha256"] == input_hash
            and item["output_sha256"] == output_hash
            for item in records
        ):
            return False
        record = {
            "case_id": case_id,
            "input_sha256": input_hash,
            "output_sha256": output_hash,
            "sequence": max((int(item["sequence"]) for item in records), default=0) + 1,
        }
        with path.open("ab", buffering=0) as handle:
            handle.write(canonical(record))
            os.fsync(handle.fileno())
    return True


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print("MIXTURE_REFUSED: expected input output ledger", file=sys.stderr)
        return 2
    input_path = Path(argv[1])
    output_path = Path(argv[2])
    ledger_path = Path(argv[3])
    before_output = output_path.read_bytes() if output_path.is_file() else None
    before_ledger = ledger_path.read_bytes() if ledger_path.is_file() else None
    try:
        payload = parse_payload(load_payload(input_path))
        report = recover(payload)
        output = canonical(report)
        atomic_write(output_path, output)
        append_ledger(
            ledger_path,
            payload["case_id"],
            digest(input_path.read_bytes()),
            digest(output),
        )
        print("MIXTURE_RECOVERED")
        return 0
    except Exception as exc:
        if before_output is None:
            try:
                output_path.unlink()
            except OSError:
                pass
        else:
            try:
                atomic_write(output_path, before_output)
            except Exception:
                pass
        if before_ledger is None:
            try:
                ledger_path.unlink()
            except OSError:
                pass
        else:
            try:
                atomic_write(ledger_path, before_ledger)
            except Exception:
                pass
        print(f"MIXTURE_REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
