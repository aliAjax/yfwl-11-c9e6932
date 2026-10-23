"""Behavior checks for censored mixture and unlabeled batch recovery."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path


CANDIDATE = os.environ.get("MIXTURE_ENTRY", "/app/mixture_recover.py")
MEANS = [-1.45, 0.15, 1.55]
SIGMAS = [0.34, 0.29, 0.42]
WEIGHTS = [
    [0.82, 0.12, 0.06],
    [0.12, 0.80, 0.08],
    [0.06, 0.12, 0.82],
    [0.55, 0.35, 0.10],
]
ASSIGN = [0, 1, 1, 2, 3, 2, 0, 3, 1, 2]


def uniform(seed: int) -> float:
    return ((seed * 1664525 + 1013904223) % (2**32)) / (2**32)


def standard_normal(seed: int) -> float:
    first = max(1e-9, uniform(seed + 101))
    second = uniform(seed + 10007)
    return math.sqrt(-2.0 * math.log(first)) * math.cos(2.0 * math.pi * second)


def canonical(value: dict) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def base_payload(case_id: str = "mixture-case", records: int = 120) -> dict:
    values = []
    for lab_index, batch in enumerate(ASSIGN):
        for item in range(12):
            seed = lab_index * 1009 + item * 9176 + 17
            pick = uniform(seed)
            weights = WEIGHTS[batch]
            component = 0 if pick < weights[0] else 1 if pick < weights[0] + weights[1] else 2
            z = standard_normal(seed + 7919 * (item + 1))
            y = MEANS[component] + SIGMAS[component] * z
            threshold = round(1.65 + 0.06 * ((lab_index + 2 * item) % 5), 6)
            values.append(
                {
                    "lab_id": f"L{lab_index + 1:02d}",
                    "value": None if y > threshold else round(y, 6),
                    "censor": threshold,
                }
            )
    return {
        "case_id": case_id,
        "max_records": 120,
        "batch_catalog": [
            {"label": "b03", "weights": WEIGHTS[2]},
            {"label": "b01", "weights": WEIGHTS[0]},
            {"label": "b04", "weights": WEIGHTS[3]},
            {"label": "b02", "weights": WEIGHTS[1]},
        ],
        "records": values[:records],
    }


def execute(
    tmp_path: Path,
    payload: dict | bytes,
    name: str,
    *,
    old_report: bytes | None = None,
    old_ledger: bytes | None = None,
    report_is_directory: bool = False,
) -> dict:
    folder = tmp_path / name
    folder.mkdir()
    source = folder / "input.json"
    report = folder / "report.json"
    ledger = folder / "ledger.jsonl"
    source.write_bytes(payload if isinstance(payload, bytes) else canonical(payload))
    if old_report is not None:
        report.write_bytes(old_report)
    if old_ledger is not None:
        ledger.write_bytes(old_ledger)
    if report_is_directory:
        report.mkdir()
    process = subprocess.run(
        [sys.executable, CANDIDATE, str(source), str(report), str(ledger)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return {"folder": folder, "source": source, "report": report, "ledger": ledger, "process": process}


def report_of(case: dict) -> dict:
    return json.loads(case["report"].read_text(encoding="utf-8"))


def ledger_rows(case: dict) -> list[dict]:
    return [
        json.loads(line)
        for line in case["ledger"].read_text(encoding="utf-8").splitlines()
        if line
    ]


def test_component_means_are_recovered_from_visible_values(tmp_path):
    """题面：「未删失值用于恢复三个组分均值，删失值作为上限证据参与质量计数。」"""
    report = report_of(execute(tmp_path, base_payload("component-means"), "case_01"))
    means = [item["mean"] for item in report["components"]]
    assert all(abs(means[index] - MEANS[index]) < 0.22 for index in range(3))


def test_component_dispersions_are_recovered(tmp_path):
    """题面：「组分尺度必须由可见样本的二阶矩恢复，不能照抄目录值。」"""
    report = report_of(execute(tmp_path, base_payload("component-sigmas"), "case_02"))
    sigmas = [item["sigma"] for item in report["components"]]
    assert all(0.12 < sigmas[index] < 0.62 for index in range(3))


def test_lab_assignments_recover_unlabeled_batches(tmp_path):
    """题面：「工位到批次的对应关系没有直接给出，必须由组分比例指纹恢复，并且四个目录标签都要至少命中一次。」"""
    report = report_of(execute(tmp_path, base_payload("lab-assignment"), "case_03"))
    actual = {item["lab_id"]: item["batch"] for item in report["lab_assignments"]}
    assert actual["L01"] == "b01"
    assert actual["L04"] == "b03"
    assert actual["L05"] == "b04"


def test_batch_profiles_are_sorted_by_label(tmp_path):
    """题面：「`batch_profiles` 按标签字典序排列，权重精确复制对应目录向量。」"""
    report = report_of(execute(tmp_path, base_payload("batch-order"), "case_04"))
    labels = [item["label"] for item in report["batch_profiles"]]
    assert labels == ["b01", "b02", "b03", "b04"]
    assert report["batch_profiles"][2]["weights"] == WEIGHTS[2]


def test_censored_and_visible_counts_are_exact(tmp_path):
    """题面：「`uncensored_count` 与 `censored_count` 必须来自输入记录，并相加等于记录总数，输入至少包含 55 条可见记录和 8 条删失记录。」"""
    payload = base_payload("counts")
    report = report_of(execute(tmp_path, payload, "case_05"))
    expected_censored = sum(item["value"] is None for item in payload["records"])
    assert report["quality"]["censored_count"] == expected_censored
    assert report["quality"]["uncensored_count"] + expected_censored == 120


def test_assignment_cost_is_reported_from_profiles(tmp_path):
    """题面：「`assignment_cost` 是十个工位比例与其命中目录向量的平方距离之和，`moment_spread` 使用同一计算口径并再次写入最终值。」"""
    report = report_of(execute(tmp_path, base_payload("assignment-cost"), "case_06"))
    assert report["quality"]["assignment_cost"] > 0.0
    assert report["quality"]["assignment_cost"] < 1.2


def test_report_is_canonical_json_with_newline(tmp_path):
    """题面：「成功报告使用 UTF-8、键排序、逗号后无空白，并以一个换行结束。」"""
    case = execute(tmp_path, base_payload("canonical-report"), "case_07")
    raw = case["report"].read_bytes()
    assert raw.endswith(b"\n")
    assert b": " not in raw
    assert raw == canonical(json.loads(raw))


def test_success_channels_use_exact_token(tmp_path):
    """题面：「成功时标准输出只写一行 `MIXTURE_RECOVERED`，标准错误为空。」"""
    case = execute(tmp_path, base_payload("success-channel"), "case_08")
    assert case["process"].returncode == 0
    assert case["process"].stdout == "MIXTURE_RECOVERED\n"
    assert case["process"].stderr == ""


def test_ledger_appends_one_record_on_success(tmp_path):
    """题面：「报告发布成功后才允许追加账本记录。」"""
    case = execute(tmp_path, base_payload("ledger-append"), "case_09")
    rows = ledger_rows(case)
    assert len(rows) == 1
    assert set(rows[0]) == {"case_id", "input_sha256", "output_sha256", "sequence"}


def test_identical_report_is_idempotent(tmp_path):
    """题面：「相同输入摘要和报告摘要的重跑必须幂等，不得追加重复记录；新的报告摘要必须按连续序号追加。」"""
    case = execute(tmp_path, base_payload("ledger-idempotent"), "case_10")
    before = case["ledger"].read_bytes()
    again = subprocess.run(
        [sys.executable, CANDIDATE, str(case["source"]), str(case["report"]), str(case["ledger"])],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert again.returncode == 0
    assert case["ledger"].read_bytes() == before


def test_changed_input_appends_next_sequence(tmp_path):
    """题面：「新的报告摘要必须按连续序号追加。」"""
    case = execute(tmp_path, base_payload("ledger-sequence"), "case_11")
    second = case["folder"] / "second.json"
    data = base_payload("ledger-sequence")
    data["records"][0]["censor"] = round(data["records"][0]["censor"] + 0.001, 6)
    second.write_bytes(canonical(data))
    process = subprocess.run(
        [sys.executable, CANDIDATE, str(second), str(case["report"]), str(case["ledger"])],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert process.returncode == 0
    assert [row["sequence"] for row in ledger_rows(case)] == [1, 2]


def test_concurrent_append_uses_exclusive_lock(tmp_path):
    """题面：「账本追加需要独占锁，中断后下一次运行从最后一个完整换行形成的检查点恢复。」"""
    folder = tmp_path / "case_12"
    folder.mkdir()
    source = folder / "input.json"
    report = folder / "report.json"
    ledger = folder / "ledger.jsonl"
    source.write_bytes(canonical(base_payload("ledger-lock")))
    processes = [
        subprocess.Popen(
            [sys.executable, CANDIDATE, str(source), str(report), str(ledger)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(3)
    ]
    completed = [process.communicate(timeout=60) for process in processes]
    assert all(process.returncode == 0 for process in processes)
    assert all(stdout == "MIXTURE_RECOVERED\n" for stdout, _ in completed)
    assert len(ledger_rows({"ledger": ledger})) == 1


def test_incomplete_ledger_tail_is_dropped(tmp_path):
    """题面：「中断后下一次运行从最后一个完整换行形成的检查点恢复。」"""
    first = canonical({"case_id":"prior","input_sha256":"a"*64,"output_sha256":"b"*64,"sequence":1})
    case = execute(
        tmp_path,
        base_payload("ledger-checkpoint"),
        "case_13",
        old_ledger=first + b'{"partial":',
    )
    assert [row["sequence"] for row in ledger_rows(case)] == [1, 2]
    assert case["ledger"].read_bytes().endswith(b"\n")


def test_refusal_preserves_old_report_and_ledger(tmp_path):
    """题面：「统计退化或发布异常发生后，值班员要看到上一版报告与账本原封不动，目录中也不残留半成品。」"""
    old_report = b'{"prior":"mixture"}\n'
    old_ledger = canonical({"case_id":"old","input_sha256":"0"*64,"output_sha256":"1"*64,"sequence":1})
    payload = base_payload("preserve")
    payload["records"] = payload["records"][:50]
    case = execute(tmp_path, payload, "case_14", old_report=old_report, old_ledger=old_ledger)
    assert case["process"].returncode == 2
    assert case["report"].read_bytes() == old_report
    assert case["ledger"].read_bytes() == old_ledger


def test_top_level_missing_key_is_refused(tmp_path):
    """题面：「批次请求分为请求号、记录上限、批次目录与观测记录四个入口。」"""
    payload = base_payload("top-missing")
    payload.pop("batch_catalog")
    case = execute(tmp_path, payload, "case_15")
    assert not case["report"].exists()
    assert not case["ledger"].exists()
    assert case["process"].returncode == 2
    assert "missing keys" in case["process"].stderr


def test_top_level_extra_key_is_refused(tmp_path):
    """题面：「批次请求分为请求号、记录上限、批次目录与观测记录四个入口。」"""
    payload = base_payload("top-extra")
    payload["notes"] = "unexpected"
    case = execute(tmp_path, payload, "case_16")
    assert not case["report"].exists()
    assert not case["ledger"].exists()
    assert case["process"].returncode == 2
    assert "extra keys" in case["process"].stderr


def test_duplicate_json_key_is_refused(tmp_path):
    """题面：「批次请求分为请求号、记录上限、批次目录与观测记录四个入口。」"""
    raw = canonical(base_payload("duplicate-top"))[:-2] + b',"case_id":"again"}\n'
    case = execute(tmp_path, raw, "case_17")
    assert not case["report"].exists()
    assert not case["ledger"].exists()
    assert case["process"].returncode == 2
    assert "duplicate key" in case["process"].stderr


def test_wrong_record_limit_is_refused(tmp_path):
    """题面：「`max_records` 必须等于公开上限 120。」"""
    payload = base_payload("wrong-max")
    payload["max_records"] = 119
    case = execute(tmp_path, payload, "case_18")
    assert not case["report"].exists()
    assert not case["ledger"].exists()
    assert case["process"].returncode == 2
    assert "max_records must equal 120" in case["process"].stderr


def test_empty_record_set_is_refused(tmp_path):
    """题面：「记录数必须在一到 120 之间。」"""
    payload = base_payload("empty-records")
    payload["records"] = []
    case = execute(tmp_path, payload, "case_19")
    assert not case["report"].exists()
    assert not case["ledger"].exists()
    assert case["process"].returncode == 2


def test_catalog_must_have_four_entries(tmp_path):
    """题面：「`batch_catalog` 恰好包含四项，每项精确键为 `label` 和 `weights`。」"""
    payload = base_payload("catalog-count")
    payload["batch_catalog"].pop()
    case = execute(tmp_path, payload, "case_20")
    assert not case["report"].exists()
    assert not case["ledger"].exists()
    assert case["process"].returncode == 2
    assert "four entries" in case["process"].stderr


def test_catalog_extra_field_is_refused(tmp_path):
    """题面：「每项精确键为 `label` 和 `weights`。」"""
    payload = base_payload("catalog-keys")
    payload["batch_catalog"][0]["extra"] = 1
    case = execute(tmp_path, payload, "case_21")
    assert not case["report"].exists()
    assert not case["ledger"].exists()
    assert case["process"].returncode == 2
    assert "extra keys" in case["process"].stderr


def test_catalog_weights_must_sum_to_one(tmp_path):
    """题面：「每条权重都严格位于零和一之间，且三项之和为一。」"""
    payload = base_payload("catalog-sum")
    payload["batch_catalog"][0]["weights"] = [0.7, 0.2, 0.2]
    case = execute(tmp_path, payload, "case_22")
    assert not case["report"].exists()
    assert not case["ledger"].exists()
    assert case["process"].returncode == 2
    assert "sum to one" in case["process"].stderr


def test_duplicate_catalog_label_is_refused(tmp_path):
    """题面：「目录标签唯一且长度为 1 到 12。」"""
    payload = base_payload("catalog-label")
    payload["batch_catalog"][1]["label"] = payload["batch_catalog"][0]["label"]
    case = execute(tmp_path, payload, "case_23")
    assert not case["report"].exists()
    assert not case["ledger"].exists()
    assert case["process"].returncode == 2
    assert "labels must be unique" in case["process"].stderr


def test_too_few_or_too_many_labs_is_refused(tmp_path):
    """题面：「记录中必须恰好出现十个工位编号，每个工位至少有四条可见记录。」"""
    payload = base_payload("lab-count")
    payload["records"] = [
        {**record, "lab_id": "L09"} if record["lab_id"] == "L10" else record
        for record in payload["records"]
    ]
    case = execute(tmp_path, payload, "case_24")
    assert not case["report"].exists()
    assert not case["ledger"].exists()
    assert case["process"].returncode == 2
    assert "exactly 10 laboratory ids" in case["process"].stderr


def test_censor_threshold_range_is_enforced(tmp_path):
    """题面：「`censor` 是 `[0.5, 2.0]` 内的有限数值。」"""
    payload = base_payload("censor-range")
    payload["records"][0]["censor"] = 2.5
    payload["records"][0]["value"] = None
    case = execute(tmp_path, payload, "case_25")
    assert not case["report"].exists()
    assert not case["ledger"].exists()
    assert case["process"].returncode == 2
    assert "censor threshold" in case["process"].stderr


def test_visible_value_above_threshold_is_refused(tmp_path):
    """题面：「可见值必须是 `[-4, 3]` 内的有限数值且不得超过其删失阈值；删失值必须写成 `null`。」"""
    payload = base_payload("visible-above")
    payload["records"][0]["value"] = payload["records"][0]["censor"] + 0.1
    case = execute(tmp_path, payload, "case_26")
    assert not case["report"].exists()
    assert not case["ledger"].exists()
    assert case["process"].returncode == 2
    assert "exceeds its censoring limit" in case["process"].stderr


def test_non_numeric_record_value_is_refused(tmp_path):
    """题面：「可见值必须是 `[-4, 3]` 内的有限数值且不得超过其删失阈值；删失值必须写成 `null`。」"""
    payload = base_payload("value-type")
    payload["records"][0]["value"] = "1.2"
    case = execute(tmp_path, payload, "case_27")
    assert not case["report"].exists()
    assert not case["ledger"].exists()
    assert case["process"].returncode == 2
    assert "must be numeric" in case["process"].stderr


def test_publish_failure_leaves_no_partial_report(tmp_path):
    """题面：「统计退化或发布异常发生后，值班员要看到上一版报告与账本原封不动，目录中也不残留半成品。」"""
    old_ledger = canonical({"case_id":"old","input_sha256":"c"*64,"output_sha256":"d"*64,"sequence":1})
    case = execute(
        tmp_path,
        base_payload("publish-failure"),
        "case_28",
        old_ledger=old_ledger,
        report_is_directory=True,
    )
    leftovers = [path.name for path in case["folder"].iterdir() if path.name.startswith(".report.json.")]
    assert case["process"].returncode == 2
    assert case["report"].is_dir()
    assert case["ledger"].read_bytes() == old_ledger
    assert leftovers == []


def test_report_key_sets_are_exact(tmp_path):
    """题面：「报告顶层精确键为 `status`、`components`、`batch_profiles`、`lab_assignments`、`quality`。」"""
    report = report_of(execute(tmp_path, base_payload("report-shape"), "case_29"))
    assert set(report) == {"status","components","batch_profiles","lab_assignments","quality"}
    assert set(report["quality"]) == {"uncensored_count","censored_count","assignment_cost","moment_spread"}


def test_candidate_entry_is_explicit_and_runnable(tmp_path):
    """题面：「调用方式是 `python3 /app/mixture_recover.py <input.json> <report.json> <ledger.jsonl>`。」"""
    case = execute(tmp_path, base_payload("entry-runnable"), "case_30")
    assert Path("/app/mixture_recover.py").as_posix() == "/app/mixture_recover.py"
    assert os.environ.get("MIXTURE_ENTRY", "/app/mixture_recover.py") == CANDIDATE
    assert case["process"].returncode == 0
