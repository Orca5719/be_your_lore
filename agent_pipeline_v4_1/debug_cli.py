"""CLI for the four-way 4.1 continuous-batching correctness experiment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid

from agent_pipeline_v2_2.runner import atomic_write_json

from .debug_experiment import (DEFAULT_NEIGHBOURS_B, DEFAULT_NEIGHBOURS_C,
                               DEFAULT_TARGET, compare_scenarios, make_scenarios,
                               run_scenario, window_request)


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "evaluation/story_benchmark_24_v2.json"
SOURCE = ROOT / "agent_pipeline_v4/reports/benchmark_4B_20261004T132545Z_0fdce1"
REPORTS = ROOT / "agent_pipeline_v4_1/reports"


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _history(case_id):
    values = {}
    for capacity in (1, 4):
        path = SOURCE / f"capacity_{capacity}" / "extractor_runs.jsonl"
        row = next((row for row in _rows(path) if row["case_id"] == case_id), None)
        if row is None:
            raise ValueError(f"historical CB{capacity} lacks {case_id}")
        extraction = row["result"]["benchmark_stages"]["extraction"]
        first = extraction["calls"][0]
        values[str(capacity)] = {
            "status": extraction["status"], "target_ids": first["target_ids"],
            "attempts": [{"status": attempt["status"], "error": attempt.get("error"),
                          "raw_sha256": hashlib.sha256(attempt.get("raw_output", "").encode("utf-8")).hexdigest()}
                         for attempt in first["attempts"]],
        }
    return values


def validate(args):
    cases = {case["id"]: case for case in _read(DATASET)["cases"]}
    scenarios = make_scenarios(args.target, tuple(args.neighbors_b), tuple(args.neighbors_c))
    unknown = set().union(*(set(order) for order in scenarios.values())) - set(cases)
    if unknown:
        raise ValueError("unknown story IDs: " + ", ".join(sorted(unknown)))
    requests = {case_id: window_request(cases[case_id])
                for case_id in set().union(*(set(order) for order in scenarios.values()))}
    history = _history(args.target)
    if history["1"]["target_ids"] != requests[args.target]["window"]["target_ids"]:
        raise ValueError("first extractor window differs from saved CB1 run")
    if history["4"]["target_ids"] != requests[args.target]["window"]["target_ids"]:
        raise ValueError("first extractor window differs from saved CB4 run")
    if history["1"]["attempts"][0]["raw_sha256"] == history["4"]["attempts"][0]["raw_sha256"]:
        raise ValueError("target first-window outputs did not differ in saved CB1 and CB4")
    return cases, scenarios, requests, history


def _identity(args, scenarios):
    from agent_pipeline_v3.model import MODEL, REVISION

    paths = [DATASET, ROOT / "agent_pipeline_v2/prompts/extractor_v1.txt",
             ROOT / "agent_pipeline_v2/extractor.py", ROOT / "agent_pipeline_v4/continuous.py",
             ROOT / "agent_pipeline_v3/model.py",
             SOURCE / "capacity_1/extractor_runs.jsonl",
             SOURCE / "capacity_4/extractor_runs.jsonl"]
    return {"schema_version": "benchmark-4.1-debug-manifest-v1",
            "target": args.target, "scenarios": scenarios, "model": MODEL,
            "model_revision": REVISION, "device": args.device,
            "max_new_tokens": 1536,
            "files": {str(path.relative_to(ROOT)).replace("\\", "/"): _sha(path) for path in paths}}


def _ensure_manifest(directory, identity):
    path = directory / "manifest.json"
    if path.exists():
        if _read(path) != identity:
            raise ValueError("debug resume identity mismatch; use a new result directory")
    else:
        atomic_write_json(path, identity)


def _summary(directory, history):
    paths = {name: directory / f"scenario_{name}.json" for name in "ABCD"}
    if not all(path.is_file() for path in paths.values()):
        print("PARTIAL_RESULT_DIR=" + str(directory), flush=True)
        return 2
    rows = {name: _read(path) for name, path in paths.items()}
    pairs = compare_scenarios(rows)
    report = {"schema_version": "benchmark-4.1-debug-summary-v1",
              "target": rows["A"]["target"], "historical_first_window": history,
              "scenarios": {name: {"admission_order": row["admission_order"],
                                  "target_initial_slot": row["target_initial_slot"],
                                  "target_input_tokens": row["target_input_tokens"],
                                  "target_output_tokens": len(row["target_tokens"]),
                                  "target_output_sha256": row["target_output_sha256"],
                                  "target_first_window_validation": row["target_first_window_validation"],
                                  "repack_count": row["repack_count"]}
                            for name, row in rows.items()},
              "first_divergence": pairs,
              "interpretation": "Neighbour/slot dependence is an observation, not proof of KV leakage; varying sequence lengths and GPU kernels can also alter floating-point logits. Inspect first divergent top-2 IDs, logits, margins, and position/cache metadata before attributing cause."}
    atomic_write_json(directory / "debug_summary.json", report)
    lines = ["# Benchmark 4.1：Continuous Batching 定位实验", "",
             f"目标：{report['target']} 第一提取窗口；A单独运行，B原同批邻居，C替换邻居，D与B同邻居但交换目标槽位。", "",
             f"历史CB1首轮状态：{history['1']['attempts'][0]['status']}；历史CB4首轮状态：{history['4']['attempts'][0]['status']}。本实验只生成首轮输出，不调用Extractor重试。", "",
             "| 组 | 目标槽位 | 同批故事（入场顺序） | 输入tokens | 输出tokens | 首次输出校验 | 输出SHA-256 |",
             "|---|---:|---|---:|---:|---|---|"]
    for name, row in rows.items():
        lines.append(f"| {name} | {row['target_initial_slot']} | {', '.join(row['admission_order'])} | "
                     f"{row['target_input_tokens']} | {len(row['target_tokens'])} | "
                     f"{row['target_first_window_validation']['status']} | {row['target_output_sha256'][:12]} |")
    lines += ["", "| 对照 | 首次分歧step | 相同前缀tokens | 左侧top-2 | 右侧top-2 |",
              "|---|---:|---:|---|---|"]
    for name, value in pairs.items():
        def top(value):
            if value is None:
                return "结束"
            return ", ".join(f"{item['token_id']}:{item['logit']:.4f}" for item in value["top2"])
        lines.append(f"| {name} | {value['first_divergence_step'] or '无'} | {value['shared_prefix_tokens']} | "
                     f"{top(value['left'])} | {top(value['right'])} |")
    lines += ["", "完整逐token logits、position/cache位置、生成文本和解析结果在各`scenario_*.json`。",
              "", "解释边界：换邻居或槽位后的差异需要结合首个分歧的logit幅度判断。序列长度变化及GPU数值路径也可能产生差异；本实验不自动宣告KV串话。",
              "", "原CB4中的邻居可能已运行若干decode步；这里控制的是相同故事的**第一窗口同时入场**，不是逐步复刻原24篇调度。若未复现原错误，结论为本受控场景未复现。", ""]
    (directory / "debug_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("SUMMARY=" + str(directory / "debug_summary.md"), flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Benchmark 4.1 four-way batching debug")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("validate-debug", "run-debug"):
        item = sub.add_parser(name)
        item.add_argument("--target", default=DEFAULT_TARGET)
        item.add_argument("--neighbors-b", nargs=3, default=DEFAULT_NEIGHBOURS_B)
        item.add_argument("--neighbors-c", nargs=3, default=DEFAULT_NEIGHBOURS_C)
        if name == "run-debug":
            item.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
            item.add_argument("--output", type=Path)
        else:
            item.set_defaults(device="cuda")
    summary = sub.add_parser("summary-debug")
    summary.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "summary-debug":
            directory = args.result_dir.resolve()
            identity = _read(directory / "manifest.json")
            saved_args = argparse.Namespace(target=identity["target"],
                                            device=identity["device"])
            if identity != _identity(saved_args, identity["scenarios"]):
                raise ValueError("debug summary source/configuration hashes changed")
            history = _history(identity["target"])
            return _summary(directory, history)
        _cases, scenarios, requests, history = validate(args)
        if args.command == "validate-debug":
            print(json.dumps({"status": "ok", "target": args.target,
                              "first_window_target_ids": requests[args.target]["window"]["target_ids"],
                              "scenarios": scenarios, "historical": history}, ensure_ascii=False, indent=2))
            return 0
        directory = args.output.resolve() if args.output else REPORTS / ("debug_4_1_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6])
        directory.mkdir(parents=True, exist_ok=True)
        _ensure_manifest(directory, _identity(args, scenarios))
        print("RESULT_DIR=" + str(directory), flush=True)
        remaining = [name for name in "ABCD" if not (directory / f"scenario_{name}.json").is_file()]
        if remaining:
            from agent_pipeline_v3.model import ProfiledV2QwenJudge

            print("正在加载一次 Qwen3 模型；随后只运行四组首窗口请求……", flush=True)
            adapter = ProfiledV2QwenJudge(args.device)
            for name in remaining:
                print(f"[{name}] {', '.join(scenarios[name])}", flush=True)
                row = run_scenario(adapter, args.target, scenarios[name], requests)
                atomic_write_json(directory / f"scenario_{name}.json", row)
        return _summary(directory, history)
    except (ValueError, RuntimeError, OSError, KeyError, json.JSONDecodeError) as exc:
        print("错误：" + str(exc), flush=True)
        return 2
