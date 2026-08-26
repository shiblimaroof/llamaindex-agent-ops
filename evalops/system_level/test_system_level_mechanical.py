"""
Tests for evalops/system_level/mechanical.py -- task_success, retries,
latency, cost. Uses real temp JSONL files (monkeypatched onto the
module's PIPELINE_LOG_PATH/USAGE_LOG_PATH), not mocks, matching the
pattern in test_error_handling_runner.py.
"""

import json
import pytest

from evalops.system_level import mechanical


def _write_jsonl(path, records):
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


@pytest.fixture
def pipeline_log(tmp_path, monkeypatch):
    path = tmp_path / "pipeline_log.jsonl"
    monkeypatch.setattr(mechanical, "PIPELINE_LOG_PATH", path)
    return path


@pytest.fixture
def usage_log(tmp_path, monkeypatch):
    path = tmp_path / "llm_usage.jsonl"
    monkeypatch.setattr(mechanical, "USAGE_LOG_PATH", path)
    return path


# ---------------------------------------------------------------------
# task_success
# ---------------------------------------------------------------------

def test_task_success_true_on_patch_application(pipeline_log):
    _write_jsonl(pipeline_log, [
        {"run_id": "r1", "node_name": "classify", "outcome": "success"},
        {"run_id": "r1", "node_name": "patch_application", "outcome": "success"},
    ])
    result = mechanical.task_success("r1")
    assert result["passed"] is True
    assert "patch_application" in result["detail"]


def test_task_success_true_on_fallback(pipeline_log):
    _write_jsonl(pipeline_log, [
        {"run_id": "r1", "node_name": "patch_application", "outcome": "failure"},
        {"run_id": "r1", "node_name": "fallback", "outcome": "success"},
    ])
    result = mechanical.task_success("r1")
    assert result["passed"] is True
    assert "fallback" in result["detail"]


def test_task_success_false_on_escalate(pipeline_log):
    _write_jsonl(pipeline_log, [
        {"run_id": "r1", "node_name": "retry", "outcome": "failure"},
        {"run_id": "r1", "node_name": "escalate", "outcome": "success"},
    ])
    result = mechanical.task_success("r1")
    assert result["passed"] is False
    assert "escalate" in result["detail"]


def test_task_success_none_when_no_terminal_node(pipeline_log):
    _write_jsonl(pipeline_log, [
        {"run_id": "r1", "node_name": "classify", "outcome": "success"},
        {"run_id": "r1", "node_name": "retrieve", "outcome": "success"},
    ])
    result = mechanical.task_success("r1")
    assert result["passed"] is None


def test_task_success_ignores_other_run_ids(pipeline_log):
    _write_jsonl(pipeline_log, [
        {"run_id": "other", "node_name": "fallback", "outcome": "success"},
    ])
    result = mechanical.task_success("r1")
    assert result["passed"] is None


# ---------------------------------------------------------------------
# retries
# ---------------------------------------------------------------------

def test_retries_excludes_bookkeeping_line(pipeline_log):
    _write_jsonl(pipeline_log, [
        {"run_id": "r1", "node_name": "retry", "duration_ms": 6843.2, "attempt": 1},
        {"run_id": "r1", "node_name": "retry", "duration_ms": 34034.1, "attempt": 2},
        {"run_id": "r1", "node_name": "retry", "duration_ms": 36134.1, "attempt": 3},
        {"run_id": "r1", "node_name": "retry", "duration_ms": 0.005, "attempt": 4},
    ])
    result = mechanical.retries("r1")
    assert result["count"] == 3
    assert "1 bookkeeping" in result["detail"]


def test_retries_zero_when_none_logged(pipeline_log):
    _write_jsonl(pipeline_log, [
        {"run_id": "r1", "node_name": "classify", "duration_ms": 600.0},
    ])
    result = mechanical.retries("r1")
    assert result["count"] == 0


def test_retries_only_counts_matching_run_id(pipeline_log):
    _write_jsonl(pipeline_log, [
        {"run_id": "r1", "node_name": "retry", "duration_ms": 100.0},
        {"run_id": "r2", "node_name": "retry", "duration_ms": 200.0},
        {"run_id": "r2", "node_name": "retry", "duration_ms": 300.0},
    ])
    result = mechanical.retries("r1")
    assert result["count"] == 1


# ---------------------------------------------------------------------
# latency
# ---------------------------------------------------------------------

def test_latency_sums_duration_ms(pipeline_log):
    _write_jsonl(pipeline_log, [
        {"run_id": "r1", "node_name": "classify", "duration_ms": 600.0},
        {"run_id": "r1", "node_name": "retrieve", "duration_ms": 19031.96},
        {"run_id": "r1", "node_name": "resolve", "duration_ms": 1953.06},
    ])
    result = mechanical.latency("r1")
    assert result["count"] == pytest.approx(21585.02, abs=0.01)
    assert "3 logged node" in result["detail"]


def test_latency_zero_when_no_matching_records(pipeline_log):
    _write_jsonl(pipeline_log, [
        {"run_id": "other", "node_name": "classify", "duration_ms": 600.0},
    ])
    result = mechanical.latency("r1")
    assert result["count"] == 0
    assert "0 logged node" in result["detail"]


def test_latency_missing_duration_field_treated_as_zero(pipeline_log):
    _write_jsonl(pipeline_log, [
        {"run_id": "r1", "node_name": "classify"},
    ])
    result = mechanical.latency("r1")
    assert result["count"] == 0


# ---------------------------------------------------------------------
# cost
# ---------------------------------------------------------------------

def test_cost_sums_across_multiple_models(usage_log):
    _write_jsonl(usage_log, [
        {
            "run_id": "r1",
            "model": "llama-3.3-70b-versatile",
            "prompt_tokens": 6382,
            "completion_tokens": 416,
        },
        {
            "run_id": "r1",
            "model": "gemini-3.6-flash",
            "prompt_tokens": 7984,
            "completion_tokens": 469,
        },
    ])
    result = mechanical.cost("r1")
    expected = (
        (6382 / 1_000_000) * 0.59 + (416 / 1_000_000) * 0.79
        + (7984 / 1_000_000) * 1.50 + (469 / 1_000_000) * 7.50
    )
    assert result["count"] == pytest.approx(expected, abs=1e-8)
    assert "2 LLM call" in result["detail"]


def test_cost_raises_on_unpriced_model(usage_log):
    _write_jsonl(usage_log, [
        {
            "run_id": "r1",
            "model": "some-unpriced-model",
            "prompt_tokens": 100,
            "completion_tokens": 50,
        },
    ])
    with pytest.raises(ValueError, match="no price entry"):
        mechanical.cost("r1")


def test_cost_zero_when_no_matching_records(usage_log):
    _write_jsonl(usage_log, [
        {"run_id": "other", "model": "llama-3.3-70b-versatile",
         "prompt_tokens": 100, "completion_tokens": 50},
    ])
    result = mechanical.cost("r1")
    assert result["count"] == 0.0
    assert "0 LLM call" in result["detail"]