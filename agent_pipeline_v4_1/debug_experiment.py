"""Controlled, first-window continuous-batching isolation experiment.

The QwenContinuousEngine implementation is reused unchanged. The experiment
controls admissions directly so neighbour identity and row order are explicit.
"""

from __future__ import annotations

from concurrent.futures import Future
import hashlib
import json
from pathlib import Path


MAX_NEW_TOKENS = 1536
DEFAULT_TARGET = "SL-010"
DEFAULT_NEIGHBOURS_B = ("SL-P04", "SL-009", "SL-P03")
DEFAULT_NEIGHBOURS_C = ("SL-011", "SL-012", "SL-013")


def make_scenarios(target: str, neighbours_b: tuple[str, ...], neighbours_c: tuple[str, ...]) -> dict[str, list[str]]:
    if (not target or len(neighbours_b) != 3 or len(neighbours_c) != 3
            or len(set(neighbours_b)) != 3 or len(set(neighbours_c)) != 3
            or target in neighbours_b or target in neighbours_c
            or set(neighbours_b) & set(neighbours_c)):
        raise ValueError("target and two non-overlapping sets of three neighbours are required")
    return {"A": [target],
            "B": [*neighbours_b, target],
            "C": [*neighbours_c, target],
            "D": [target, *neighbours_b]}


def first_divergence(left: list[dict], right: list[dict]) -> dict:
    for index in range(max(len(left), len(right))):
        a = left[index] if index < len(left) else None
        b = right[index] if index < len(right) else None
        if a is None or b is None or a["token_id"] != b["token_id"]:
            return {"same_token_sequence": False, "first_divergence_step": index + 1,
                    "left": a, "right": b, "shared_prefix_tokens": index}
    return {"same_token_sequence": True, "first_divergence_step": None,
            "left": None, "right": None, "shared_prefix_tokens": len(left)}


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def window_request(case: dict) -> dict:
    """Reconstruct exactly the first extractor request, without retry prompts."""
    from agent_pipeline_v2.extractor import split_spans, _windows

    spans = split_spans(case["story"])
    window = _windows(spans)[0]
    prompt = (Path(__file__).resolve().parents[1] / "agent_pipeline_v2/prompts/extractor_v1.txt").read_text(encoding="utf-8")
    payload = {
        "target_spans": {sid: spans[sid]["text"] for sid in window["target_ids"]},
        "context_spans": {sid: spans[sid]["text"] for sid in window["context_ids"]},
    }
    messages = [{"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}]
    return {"messages": messages, "spans": spans, "window": window,
            "prompt_sha256": sha256_text(json.dumps(messages, ensure_ascii=False, sort_keys=True))}


class Top2Probe:
    """Read model logits with a forward hook; never change outputs or decoding."""

    def __init__(self, model):
        self.model = model
        self._pending: list[list[list[dict]]] = []
        self._handle = None

    def __enter__(self):
        self._handle = self.model.register_forward_hook(self._capture)
        return self

    def __exit__(self, *_):
        self._handle.remove()

    def _capture(self, _model, _inputs, output):
        logits = output.logits[:, -1, :]
        values, indices = logits.topk(2, dim=-1)
        values = values.detach().float().cpu().tolist()
        indices = indices.detach().cpu().tolist()
        self._pending.append([
            [{"token_id": int(token), "logit": float(value)} for token, value in zip(ids, scores)]
            for ids, scores in zip(indices, values)
        ])

    def take(self, expected_rows: int) -> list[list[dict]]:
        if len(self._pending) != 1 or len(self._pending[0]) != expected_rows:
            raise RuntimeError(f"expected one model forward with {expected_rows} rows, got {len(self._pending)}")
        return self._pending.pop()


def _entry(step: int, token: int, top2: list[dict], *, active_order: list[str],
           cache_position: int | None, position_id: int | None, repack_count: int) -> dict:
    if int(top2[0]["token_id"]) != int(token):
        raise RuntimeError("observed top-1 logit does not match generated token")
    return {"step": step, "token_id": int(token), "top2": top2,
            "top2_margin": top2[0]["logit"] - top2[1]["logit"],
            "active_order": active_order, "cache_position": cache_position,
            "position_id": position_id, "repack_count": repack_count}


def _decode_target(raw: str, spec: dict, *, complete: bool) -> dict:
    from agent_pipeline_v2.extractor import _decode_window, _normalize_window_value

    try:
        if not complete:
            raise ValueError("generation reached max_new_tokens without EOS")
        value = json.loads(raw)
        normalized, changes = _normalize_window_value(value, spec["spans"], spec["window"])
        events, ignored, non_events = _decode_window(normalized, spec["spans"], spec["window"], 1)
        return {"status": "ok", "events": events, "ignored_spans": ignored,
                "non_event_span_ids": non_events, "normalizations": changes}
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


def run_scenario(adapter, target: str, order: list[str], requests: dict[str, dict]) -> dict:
    """Run independent first-window calls with deterministic admission order."""
    from agent_pipeline_v4.continuous import QwenContinuousEngine, _Request

    engine = QwenContinuousEngine(adapter)
    active: list[dict] = []
    outputs: dict[str, list[int]] = {}
    traces: dict[str, list[dict]] = {}
    ended: dict[str, bool] = {}
    with Top2Probe(adapter.model) as probe:
        for case_id in order:
            spec = requests[case_id]
            request = _Request(case_id, spec["messages"], MAX_NEW_TOKENS, Future())
            state, token, metrics = engine.prefill(request)
            top2 = probe.take(1)[0]
            outputs[case_id] = [int(token)]
            traces[case_id] = [_entry(1, token, top2, active_order=[case_id],
                                      cache_position=None, position_id=int(metrics["input_tokens"])-1,
                                      repack_count=engine.repack_count)]
            ended[case_id] = token in engine.eos_ids
            if not ended[case_id]:
                active.append({"id": case_id, "state": state})
        while active:
            names = [row["id"] for row in active]
            states = [row["state"] for row in active]
            positions = [int(state["mask"].sum().item()) for state in states]
            cache_position = max(int(state["mask"].shape[-1]) for state in states)
            tokens, _metrics = engine.decode(states, [outputs[name][-1] for name in names])
            tops = probe.take(len(active))
            survivors = []
            for row, token, top2, position in zip(active, tokens, tops, positions):
                name = row["id"]
                outputs[name].append(int(token))
                traces[name].append(_entry(len(outputs[name]), token, top2,
                                           active_order=names, cache_position=cache_position,
                                           position_id=position, repack_count=engine.repack_count))
                done = token in engine.eos_ids
                ended[name] = done
                if not done and len(outputs[name]) < MAX_NEW_TOKENS:
                    survivors.append(row)
            active = survivors
    target_tokens = outputs[target]
    target_raw = engine.text(target_tokens)
    return {
        "schema_version": "benchmark-4.1-debug-scenario-v1",
        "target": target, "admission_order": order, "target_initial_slot": order.index(target),
        "target_prompt_sha256": requests[target]["prompt_sha256"],
        "target_input_tokens": traces[target][0]["position_id"] + 1,
        "target_tokens": target_tokens, "target_trace": traces[target],
        "target_output": target_raw, "target_output_sha256": sha256_text(target_raw),
        "target_finished_with_eos": ended[target],
        "target_first_window_validation": _decode_target(target_raw, requests[target], complete=ended[target]),
        "neighbours": {name: {"output_tokens": len(outputs[name]), "token_sha256": sha256_text(json.dumps(outputs[name]))}
                       for name in order if name != target},
        "repack_count": engine.repack_count,
    }


def compare_scenarios(rows: dict[str, dict]) -> dict:
    expected = {"A", "B", "C", "D"}
    if set(rows) != expected:
        raise ValueError("all A/B/C/D scenario results are required")
    prompts = {row["target_prompt_sha256"] for row in rows.values()}
    if len(prompts) != 1:
        raise ValueError("target prompt changed across scenarios")
    pairs = (("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D"))
    return {f"{left}-{right}": first_divergence(rows[left]["target_trace"], rows[right]["target_trace"])
            for left, right in pairs}
