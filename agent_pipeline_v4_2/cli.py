"""Benchmark 4.2A: paired serial Extractor decoding experiment."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid

from agent_pipeline_v2_2.runner import atomic_write_json, write_jsonl_atomic
from agent_pipeline_v4.cli import _quality_and_performance, validate_inputs
from agent_pipeline_v4.static import run_static_cases

from .model import ConstrainedExtractorModel, extractor_json_schema
from .quality import build_review_template, score_reviewed


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "agent_pipeline_v4_2" / "reports"
MODES = ("baseline", "constrained")


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _identity(inputs: dict) -> dict:
    import importlib.metadata

    schema_bytes = json.dumps(extractor_json_schema(), sort_keys=True, ensure_ascii=False).encode("utf-8")
    implementation = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
        "agent_pipeline_v4_2/model.py", "agent_pipeline_v4_2/cli.py",
        "agent_pipeline_v4_2/quality.py", "uv.lock",
        "agent_pipeline_v2_1/story_pipeline.py", "agent_pipeline_v2_1/judge.py",
        "agent_pipeline_v2_1/retrieval.py", "agent_pipeline_v2_1/prompts/judge_v2_1.txt",
        "agent_pipeline_v2_2/systems.py", "data/index/CURRENT",
    )}
    index_version = (ROOT / "data/index/CURRENT").read_text(encoding="ascii").strip()
    for name in ("manifest.json", "metadata.json"):
        relative = f"data/index/versions/{index_version}/{name}"
        implementation[relative] = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
    return {
        "schema_version": "benchmark-4.2A-manifest-v1",
        "source": inputs["identity"],
        "grammar_sha256": hashlib.sha256(schema_bytes).hexdigest(),
        "implementation": implementation,
        "xgrammar_version": importlib.metadata.version("xgrammar"),
        "modes": list(MODES),
        "extractor_batch_size": 1,
        "warmup_stories_per_mode": 1,
    }


def _ensure_manifest(directory: Path, identity: dict) -> None:
    path = directory / "manifest.json"
    if path.exists():
        if _read(path) != identity:
            raise ValueError("4.2A 输入、模型或依赖版本变化；请使用新结果目录")
    else:
        atomic_write_json(path, identity)


def _result_dir(output: Path | None) -> Path:
    if output:
        return output.resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return REPORTS / f"benchmark_4_2A_{stamp}_{uuid.uuid4().hex[:6]}"


def _fmt(value: float | None, digits: int = 2) -> str:
    return "待复核" if value is None else f"{value:.{digits}f}"


def summarize(directory: Path, inputs: dict) -> dict:
    _ensure_manifest(directory, _identity(inputs))
    arms: dict[str, dict] = {}
    for mode in MODES:
        path = directory / mode
        if not (path / "extractor_runs.jsonl").exists():
            continue
        previous = _read(path / "summary.json") if (path / "summary.json").exists() else {}
        wall = previous.get("performance", {}).get("wall_seconds")
        arms[mode] = _quality_and_performance(path, 1, inputs, wall)
        failures = Counter()
        for row in _jsonl(path / "extractor_runs.jsonl"):
            extraction = row["result"]["benchmark_stages"]["extraction"]
            for window in extraction.get("calls", []):
                for attempt in window.get("attempts", []):
                    if attempt.get("status") == "error":
                        failures[attempt.get("error_type") or "unknown"] += 1
        arms[mode]["failed_attempts_by_error_type"] = dict(sorted(failures.items()))
        review_path = path / "review.json"
        extraction_rows = _jsonl(path / "extractor_runs.jsonl")
        if not review_path.exists() and len(extraction_rows) == len(inputs["cases"]):
            review = build_review_template(inputs["source_rows"], extraction_rows, inputs["review"], mode)
            review["extraction_sha256"] = hashlib.sha256((path / "extractor_runs.jsonl").read_bytes()).hexdigest()
            atomic_write_json(review_path, review)
    result = {"schema_version": "benchmark-4.2A-summary-v1", "arms": arms,
              "quality_notice": "改变的事件须逐案审核；pending 时 Recall/Precision 为空，不能用下界冒充正式分数。",
              "performance_notice": "恢复运行的 Wall 耗时不可与完整单次运行横向比较。"}
    metadata = directory / "run_metadata.json"
    if metadata.exists():
        result["setup"] = _read(metadata)
    atomic_write_json(directory / "benchmark_4_2A_summary.json", result)
    lines = ["# Benchmark 4.2A：Extractor 约束解码", "",
             "同一次运行内顺序对照；两组均为 batch=1、原 Extractor prompt/schema 与一次预热。只改变生成时是否使用 JSON logits 约束。", "",
             "| 组别 | 故事状态 ok/partial/error | 调用 | 格式重试/覆盖恢复 | 输入/输出 tokens | TTFT p50/p95 ms | Prefill/Decode/Wall s | Decode tok/s | 峰值 allocated/reserved GiB | Extraction Recall/Precision | 待审核事件 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    if metadata.exists():
        setup = _read(metadata)
        lines[3:3] = [f"模型加载 {setup['model_load_seconds']:.2f} s；语法编译 {setup['grammar_compile_seconds']:.2f} s，均未计入下表推理 Wall。", ""]
    for mode in MODES:
        item = arms.get(mode)
        if not item:
            continue
        perf, quality = item["performance"], item["quality"]
        counts = perf["status_counts"]
        q = quality["candidate"]
        lines.append("| " + " | ".join([
            mode,
            f"{counts['ok']}/{counts['partial']}/{counts['error']}",
            str(perf["llm_calls"]),
            f"{perf['retry_requests']}/{perf['coverage_recovery_requests']}",
            f"{perf['input_tokens']}/{perf['output_tokens']}",
            f"{_fmt(perf['ttft_p50_ms'])}/{_fmt(perf['ttft_p95_ms'])}",
            f"{_fmt(perf['prefill_seconds'])}/{_fmt(perf['decode_seconds'])}/{_fmt(perf['wall_seconds'])}",
            _fmt(perf["decode_tokens_per_second"]),
            f"{_fmt(perf['peak_allocated'] / 2**30 if perf['peak_allocated'] is not None else None)}/{_fmt(perf['peak_reserved'] / 2**30 if perf['peak_reserved'] is not None else None)}",
            f"{_fmt(q['recall'] * 100 if q['recall'] is not None else None)}%/{_fmt(q['precision'] * 100 if q['precision'] is not None else None)}%",
            str(quality["pending_count"]),
        ]) + " |")
    if len(arms) == 2:
        base = arms["baseline"]["performance"]
        candidate = arms["constrained"]["performance"]
        if base["wall_seconds"] and candidate["wall_seconds"]:
            lines += ["", f"Extractor Wall 比值（baseline/constrained）：{base['wall_seconds'] / candidate['wall_seconds']:.3f}x。"]
    lines += ["", "失败尝试（按原异常类型，包含格式、字段和引用等错误；覆盖恢复另见上表）："]
    for mode in MODES:
        if mode in arms:
            lines.append(f"- {mode}: " + json.dumps(arms[mode]["failed_attempts_by_error_type"], ensure_ascii=False))
    lines += ["", "JSON 合法不等于事实正确；相似度也不是矛盾置信度。新/改变事件在各组 pending_event_review.json 中。", "",
              "本表为 Extractor-only。端到端 F1 须在事件审核后运行冻结的检索、Judge 和 Report，不能把本表当成端到端成绩。", ""]
    summary = directory / "benchmark_4_2A_summary.md"
    summary.write_text("\n".join(lines), encoding="utf-8")
    print("SUMMARY=" + str(summary), flush=True)
    return result


def run_downstream(args) -> int:
    """Re-run the frozen downstream only after every changed event is reviewed."""
    inputs = validate_inputs()
    directory = args.result_dir.resolve()
    _ensure_manifest(directory, _identity(inputs))
    runs = {}
    for mode in MODES:
        source = directory / mode / "extractor_runs.jsonl"
        review = _read(directory / mode / "review.json")
        if review.get("pending") or review.get("extraction_sha256") != hashlib.sha256(source.read_bytes()).hexdigest():
            raise ValueError(f"{mode} 审核未完成或提取结果已改变")
        rows = _jsonl(source)
        from agent_pipeline_v4_1.audit import validate_labels
        validate_labels(inputs["cases"], rows, review["cases"])
        runs[mode] = (rows, review)

    from agent_pipeline_v3.model import ProfiledV2QwenJudge
    from agent_pipeline_v2_1.retrieval import V21Retriever
    from agent_pipeline_v4_1.cli import _run_one
    from encoder import Encoder
    from retrieval import Retriever

    llm = ProfiledV2QwenJudge(args.device)
    dense = Retriever(ROOT / "data/index", Encoder(device=llm.device, offline=True, precision="float32"))
    metadata = _read(ROOT / "evaluation/agent_pipeline_v2_1_lore_metadata.json")
    retriever = V21Retriever(dense, metadata)
    scores = {}
    for mode in MODES:
        extraction_rows, review = runs[mode]
        by_id = {row["case_id"]: row for row in extraction_rows}
        path = directory / mode / "downstream_runs.jsonl"
        downstream = _jsonl(path)
        seen = {row["case_id"] for row in downstream}
        if len(seen) != len(downstream):
            raise ValueError(f"{mode} 下游结果存在重复故事")
        for case in inputs["cases"]:
            cid = case["id"]
            if cid in seen:
                continue
            llm.set_story_id(mode + ":" + cid)
            print(f"[{mode} downstream {len(downstream)+1}/24] {cid}", flush=True)
            extraction = by_id[cid]["result"]["benchmark_stages"]["extraction"]
            result = _run_one(extraction, llm, retriever)
            downstream.append({"case_id": cid, "system": mode, **result})
            write_jsonl_atomic(path, downstream)
            llm.clear_traces()
        scores[mode] = score_reviewed(inputs["cases"], extraction_rows, downstream, review)
    atomic_write_json(directory / "quality_reviewed.json", scores)
    lines = ["# Benchmark 4.2A：审核后质量", "",
             "两组使用相同的冻结 Hybrid Retrieval、Judge v2.1（batch 8）和 Report；提取事件已按 review.json 审核。", "",
             "| 组别 | Extraction Recall | Extraction Precision | Hallucination | Overselect | E2E TP/FP/FN | E2E F1 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for mode in MODES:
        q = scores[mode]["extraction"]
        e = scores[mode]["end_to_end"]
        lines.append(f"| {mode} | {q['recall']:.2%} | {q['precision']:.2%} | "
                     f"{q['hallucinated_events']}/{q['emitted_events']} | "
                     f"{q['overselected_events']}/{q['emitted_events']} | "
                     f"{e['tp']}/{e['fp']}/{e['fn']} | {e['f1']:.2%} |")
    report = directory / "quality_reviewed.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("QUALITY=" + str(report), flush=True)
    return 0


def run(args) -> int:
    inputs = validate_inputs()
    directory = _result_dir(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    _ensure_manifest(directory, _identity(inputs))
    print("RESULT_DIR=" + str(directory), flush=True)
    print("正在加载一次 Benchmark 3 Qwen 模型并编译 Extractor JSON 语法……", flush=True)
    model = ConstrainedExtractorModel(args.device)
    metadata = directory / "run_metadata.json"
    if not metadata.exists():
        atomic_write_json(metadata, {"model_load_seconds": model.load_seconds,
                                     "grammar_compile_seconds": model.grammar_compile_seconds,
                                     "mask_path": "XGrammar compiled grammar; CPU bitmask and CUDA model logits transfer on Windows"})
    for mode in MODES:
        output = directory / mode
        previous = _jsonl(output / "extractor_runs.jsonl")
        if (output / "review.json").exists() and any(row.get("status") != "ok" for row in previous):
            raise ValueError(f"{mode} 已建立审核账本且含未完成故事；请使用新结果目录重跑，避免覆盖已审核内容")
        model.constraint_enabled = mode == "constrained"
        model.set_story_id(None)
        print(f"[{mode}] 预热后运行 24 篇故事", flush=True)
        measured = run_static_cases(inputs["cases"], model, 1, output,
                                    warmup=True,
                                    progress=lambda done, total, cid: print(f"  [{mode}] {done}/{total} {cid}", flush=True))
        wall = measured["wall_seconds"]
        if previous and len(previous) < len(inputs["cases"]):
            wall = None
        elif wall is None and (output / "summary.json").exists():
            wall = _read(output / "summary.json")["performance"]["wall_seconds"]
        _quality_and_performance(output, 1, inputs, wall)
        model.clear_traces()
        summarize(directory, inputs)
    final = summarize(directory, inputs)
    return 0 if all(item["complete"] for item in final["arms"].values()) and len(final["arms"]) == 2 else 2


def smoke(args) -> int:
    """Small real-model check; never counted in the formal report."""
    from agent_pipeline_v2.extractor import extract_events
    from agent_pipeline_v2_1.extractor import ExtractionRepairLLM
    from agent_pipeline_v4.static import QueuedLLM, StaticBatcher

    model = ConstrainedExtractorModel(args.device)
    story = "雷捂住右胸，感到寄宿其中的亚巴顿开始躁动。"
    results = {}
    for mode in MODES:
        model.constraint_enabled = mode == "constrained"
        model.set_story_id("SMOKE:" + mode)
        with StaticBatcher(model, 1) as batcher:
            report = extract_events(story, device=model.device,
                                    llm=ExtractionRepairLLM(QueuedLLM(batcher, "SMOKE:" + mode)))
        results[mode] = {"status": report["status"], "events": len(report["events"]),
                         "calls": sum(len(call["attempts"]) for call in report["calls"]),
                         "uncovered_spans": report.get("uncovered_span_ids", [])}
        model.clear_traces()
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)
    return 0 if all(value["status"] == "ok" for value in results.values()) else 2


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark 4.2A constrained Extractor generation")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate")
    smoke_parser = sub.add_parser("smoke-a")
    smoke_parser.add_argument("--device", choices=("cuda",), default="cuda")
    run_parser = sub.add_parser("run-a")
    run_parser.add_argument("--device", choices=("cuda",), default="cuda")
    run_parser.add_argument("--output", type=Path)
    summary_parser = sub.add_parser("summary")
    summary_parser.add_argument("--result-dir", type=Path, required=True)
    downstream_parser = sub.add_parser("run-downstream")
    downstream_parser.add_argument("--result-dir", type=Path, required=True)
    downstream_parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args(argv)
    try:
        inputs = validate_inputs()
        if args.command == "validate":
            import importlib.metadata
            print(json.dumps({"stories": len(inputs["cases"]), "model": inputs["identity"]["model"],
                              "revision": inputs["identity"]["revision"], "experiment": "4.2A",
                              "xgrammar": importlib.metadata.version("xgrammar")},
                             ensure_ascii=False, indent=2))
            return 0
        if args.command == "run-a":
            return run(args)
        if args.command == "smoke-a":
            return smoke(args)
        if args.command == "run-downstream":
            return run_downstream(args)
        result = summarize(args.result_dir.resolve(), inputs)
        return 0 if result["arms"] else 2
    except (ValueError, OSError, RuntimeError, KeyError, ImportError) as exc:
        parser.exit(2, f"错误：{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
