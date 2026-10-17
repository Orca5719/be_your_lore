"""Benchmark 4.2B: prompt-lookup speculative decoding of the frozen Extractor."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid

from agent_pipeline_v2_2.runner import atomic_write_json, write_jsonl_atomic
from agent_pipeline_v4.cli import _quality_and_performance, validate_inputs
from agent_pipeline_v4.static import run_static_cases
from agent_pipeline_v4_1.audit import score_extraction, validate_labels

from .quality import build_review_template, score_reviewed
from .speculative_model import PromptLookupExtractorModel


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "agent_pipeline_v4_2" / "reports"
BASELINE_RESULT = REPORTS / "benchmark_4_2A_20261016T170530Z_ab70a2"


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _baseline_files(baseline: Path) -> dict[str, Path]:
    return {
        "baseline_extractions": baseline / "baseline/extractor_runs.jsonl",
        "baseline_calls": baseline / "baseline/inference_calls.jsonl",
        "baseline_review": baseline / "baseline/review.json",
        "baseline_downstream": baseline / "baseline/downstream_runs.jsonl",
        "baseline_quality": baseline / "quality_reviewed.json",
    }


def validate_b(baseline: Path) -> dict:
    inputs = validate_inputs()
    required = _baseline_files(baseline)
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise ValueError("4.2B 缺少已完成的无约束 batch=1 基线：" + ", ".join(missing))
    rows = _jsonl(required["baseline_extractions"])
    if len(rows) != 24 or {r["case_id"] for r in rows} != {c["id"] for c in inputs["cases"]}:
        raise ValueError("4.2B 基线不是完整的24篇")
    if any(row["status"] != "ok" for row in rows):
        raise ValueError("4.2B 基线包含未成功的故事")
    review = _read(required["baseline_review"])
    if review.get("pending"):
        raise ValueError("4.2B 基线仍有未审核事件")
    validate_labels(inputs["cases"], rows, review["cases"])
    if len(_jsonl(required["baseline_downstream"])) != 24:
        raise ValueError("4.2B 基线下游不完整")
    return inputs


def _identity(inputs: dict, baseline: Path, num_tokens: int) -> dict:
    import importlib.metadata

    files = {name: _sha(path) for name, path in _baseline_files(baseline).items()}
    files["speculative_model"] = _sha(ROOT / "agent_pipeline_v4_2/speculative_model.py")
    return json.loads(json.dumps({
        "schema_version": "benchmark-4.2B-manifest-v1",
        "source": inputs["identity"],
        "baseline_files": files,
        "baseline_result": str(baseline.resolve()),
        "transformers_version": importlib.metadata.version("transformers"),
        "prompt_lookup_num_tokens": num_tokens,
        "extractor_batch_size": 1,
        "warmup_stories": 1,
        "generation": "Benchmark-3 greedy NF4 BF16, unchanged Extractor prompt/schema",
    }, ensure_ascii=False))


def _ensure_manifest(directory: Path, identity: dict):
    path = directory / "manifest.json"
    if path.exists():
        if _read(path) != identity:
            raise ValueError("4.2B 输入、模型、基线或生成配置变化；请使用新结果目录")
    else:
        atomic_write_json(path, identity)


def _result_dir(output: Path | None) -> Path:
    if output:
        return output.resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return REPORTS / f"benchmark_4_2B_{stamp}_{uuid.uuid4().hex[:6]}"


def _fmt(value, digits=2):
    return "待复核" if value is None else f"{value:.{digits}f}"


def summarize(directory: Path, inputs: dict, baseline: Path, num_tokens: int) -> dict:
    _ensure_manifest(directory, _identity(inputs, baseline, num_tokens))
    output = directory / "prompt_lookup"
    source = output / "extractor_runs.jsonl"
    if not source.exists():
        raise ValueError("尚无 4.2B 推测解码结果")
    previous = _read(output / "summary.json") if (output / "summary.json").exists() else {}
    wall = previous.get("performance", {}).get("wall_seconds")
    summary = _quality_and_performance(output, 1, inputs, wall)
    rows = _jsonl(source)
    review_path = output / "review.json"
    if not review_path.exists() and len(rows) == 24:
        review = build_review_template(inputs["source_rows"], rows, inputs["review"], "prompt_lookup")
        review["extraction_sha256"] = _sha(source)
        atomic_write_json(review_path, review)
    reviewed = None
    if review_path.exists():
        review = _read(review_path)
        if not review.get("pending") and review.get("extraction_sha256") == _sha(source):
            reviewed = score_extraction(inputs["cases"], rows, review["cases"])
    base_quality = _read(baseline / "quality_reviewed.json")["baseline"]
    base_perf = _read(baseline / "baseline/summary.json")["performance"]
    perf = summary["performance"]
    calls = _jsonl(output / "inference_calls.jsonl")
    steps = sum(int(call.get("commit_steps", 0)) for call in calls)
    multi = sum(int(call.get("multi_token_steps", 0)) for call in calls)
    extra = sum(int(call.get("extra_committed_tokens", 0)) for call in calls)
    result = {
        "schema_version": "benchmark-4.2B-summary-v1",
        "baseline": {"quality": base_quality, "performance": base_perf},
        "prompt_lookup": {"extractor": summary, "reviewed_quality": reviewed},
        "generation_steps": {"commit_steps": steps, "multi_token_steps": multi,
                             "extra_committed_tokens": extra,
                             "tokens_per_commit_step": perf["output_tokens"] / steps if steps else None},
        "comparison_notice": "无约束基线复用4.2A已保存结果，跨轮比较；batch均为1，不能与batch=4/8吞吐混作同口径。",
    }
    setup = directory / "run_metadata.json"
    if setup.exists():
        result["setup"] = _read(setup)
    atomic_write_json(directory / "benchmark_4_2B_summary.json", result)
    candidate = reviewed or summary["quality"]["candidate"]
    qbase = base_quality["extraction"]
    lines = ["# Benchmark 4.2B：Prompt Lookup 推测解码", "",
             "同一 Qwen3-4B、Extractor prompt/schema、greedy生成和batch=1；只增加 `prompt_lookup_num_tokens`。无约束结果来自4.2A已完成的baseline，性能为跨轮对照。", "",
             "| 组别 | 故事 ok/partial/error | 调用 | 重试/覆盖恢复 | 输入/输出 tokens | TTFT p50/p95 ms | Prefill/Decode/Wall s | Decode tok/s | 峰值 allocated/reserved GiB | Extraction Recall/Precision | 待审核 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, data, quality in (("baseline", base_perf, qbase), ("prompt_lookup", perf, candidate)):
        counts = data["status_counts"]
        lines.append("| " + " | ".join([
            name, f"{counts['ok']}/{counts['partial']}/{counts['error']}",
            str(data["llm_calls"]), f"{data['retry_requests']}/{data['coverage_recovery_requests']}",
            f"{data['input_tokens']}/{data['output_tokens']}",
            f"{_fmt(data['ttft_p50_ms'])}/{_fmt(data['ttft_p95_ms'])}",
            f"{_fmt(data['prefill_seconds'])}/{_fmt(data['decode_seconds'])}/{_fmt(data['wall_seconds'])}",
            _fmt(data["decode_tokens_per_second"]),
            f"{_fmt(data['peak_allocated']/2**30 if data['peak_allocated'] is not None else None)}/{_fmt(data['peak_reserved']/2**30 if data['peak_reserved'] is not None else None)}",
            f"{_fmt(quality['recall']*100 if quality['recall'] is not None else None)}%/{_fmt(quality['precision']*100 if quality['precision'] is not None else None)}%",
            "0" if name == "baseline" or reviewed else str(summary["quality"]["pending_count"]),
        ]) + " |")
    lines += ["", f"Prompt Lookup 候选长度：{num_tokens} tokens；共 {steps} 个提交步，{multi} 步一次提交多个token，"
              f"额外提交 {extra} tokens；平均每步提交 {_fmt(perf['output_tokens']/steps if steps else None)} tokens。",
              "这些是streamer观察到的提交量，并非草稿模型接受率；本实验没有辅助模型。", "",
              "若新增事件仍待审核，Recall/Precision不发布正式值；下游E2E F1需审核后另跑。", "",
              "Transformers 4.57.6 的辅助生成路径只接受batch=1；本轮评估单请求路径，不声称多请求吞吐收益。", ""]
    report = directory / "benchmark_4_2B_summary.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print("SUMMARY=" + str(report), flush=True)
    return result


def run(args) -> int:
    baseline = args.baseline_result.resolve()
    inputs = validate_b(baseline)
    directory = _result_dir(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    _ensure_manifest(directory, _identity(inputs, baseline, args.prompt_lookup_num_tokens))
    print("RESULT_DIR=" + str(directory), flush=True)
    print("正在加载一次Benchmark 3 Qwen模型并运行Prompt Lookup提取……", flush=True)
    model = PromptLookupExtractorModel(args.device, num_tokens=args.prompt_lookup_num_tokens)
    setup = directory / "run_metadata.json"
    if not setup.exists():
        atomic_write_json(setup, {"model_load_seconds": model.load_seconds})
    output = directory / "prompt_lookup"
    previous = _jsonl(output / "extractor_runs.jsonl")
    if (output / "review.json").exists() and any(row.get("status") != "ok" for row in previous):
        raise ValueError("已有审核账本和未完成故事；请使用新结果目录")
    measured = run_static_cases(inputs["cases"], model, 1, output, warmup=True,
                                progress=lambda done, total, cid: print(f"[prompt_lookup] {done}/{total} {cid}", flush=True))
    wall = measured["wall_seconds"]
    if previous and len(previous) < len(inputs["cases"]):
        wall = None
    elif wall is None and (output / "summary.json").exists():
        wall = _read(output / "summary.json")["performance"]["wall_seconds"]
    _quality_and_performance(output, 1, inputs, wall)
    result = summarize(directory, inputs, baseline, args.prompt_lookup_num_tokens)
    return 0 if result["prompt_lookup"]["extractor"]["complete"] else 2


def smoke(args) -> int:
    from agent_pipeline_v2.extractor import extract_events
    from agent_pipeline_v2_1.extractor import ExtractionRepairLLM
    from agent_pipeline_v4.static import QueuedLLM, StaticBatcher

    model = PromptLookupExtractorModel(args.device, num_tokens=args.prompt_lookup_num_tokens)
    text = "雷捂住右胸，感到寄宿其中的亚巴顿开始躁动。"
    results = {}
    for name, enabled in (("baseline", False), ("prompt_lookup", True)):
        model.enabled = enabled
        with StaticBatcher(model, 1) as batcher:
            report = extract_events(text, device=model.device,
                                    llm=ExtractionRepairLLM(QueuedLLM(batcher, "SMOKE:" + name)))
        results[name] = {"status": report["status"], "events": report["events"],
                         "calls": len(batcher.records),
                         "commit_steps": sum(r.get("commit_steps", 0) for r in batcher.records)}
        model.clear_traces()
    same = results["baseline"]["events"] == results["prompt_lookup"]["events"]
    print(json.dumps({"baseline": results["baseline"], "prompt_lookup": results["prompt_lookup"],
                      "events_identical": same}, ensure_ascii=False, indent=2), flush=True)
    return 0 if all(results[name]["status"] == "ok" for name in results) else 2


def run_downstream(args) -> int:
    baseline = args.baseline_result.resolve()
    inputs = validate_b(baseline)
    directory = args.result_dir.resolve()
    identity = _read(directory / "manifest.json")
    _ensure_manifest(directory, _identity(inputs, baseline, identity["prompt_lookup_num_tokens"]))
    output = directory / "prompt_lookup"
    source = output / "extractor_runs.jsonl"
    review = _read(output / "review.json")
    if review.get("pending") or review.get("extraction_sha256") != _sha(source):
        raise ValueError("推测解码事件审核未完成或提取结果已改变")
    rows = _jsonl(source)
    validate_labels(inputs["cases"], rows, review["cases"])

    from agent_pipeline_v3.model import ProfiledV2QwenJudge
    from agent_pipeline_v2_1.retrieval import V21Retriever
    from agent_pipeline_v4_1.cli import _run_one
    from encoder import Encoder
    from retrieval import Retriever

    llm = ProfiledV2QwenJudge(args.device)
    dense = Retriever(ROOT / "data/index", Encoder(device=llm.device, offline=True, precision="float32"))
    metadata = _read(ROOT / "evaluation/agent_pipeline_v2_1_lore_metadata.json")
    retriever = V21Retriever(dense, metadata)
    path = output / "downstream_runs.jsonl"
    downstream = _jsonl(path)
    seen = {row["case_id"] for row in downstream}
    if len(seen) != len(downstream):
        raise ValueError("下游结果存在重复故事")
    by_id = {row["case_id"]: row for row in rows}
    for case in inputs["cases"]:
        cid = case["id"]
        if cid in seen:
            continue
        llm.set_story_id("prompt_lookup:" + cid)
        print(f"[prompt_lookup downstream {len(downstream)+1}/24] {cid}", flush=True)
        extraction = by_id[cid]["result"]["benchmark_stages"]["extraction"]
        result = _run_one(extraction, llm, retriever)
        downstream.append({"case_id": cid, "system": "prompt_lookup", **result})
        write_jsonl_atomic(path, downstream)
        llm.clear_traces()
    candidate = score_reviewed(inputs["cases"], rows, downstream, review)
    baseline_quality = _read(baseline / "quality_reviewed.json")["baseline"]
    scores = {"baseline": baseline_quality, "prompt_lookup": candidate}
    atomic_write_json(directory / "quality_reviewed.json", scores)
    lines = ["# Benchmark 4.2B：审核后质量", "",
             "冻结Hybrid Retrieval、Judge v2.1 batch=8和Report；无约束基线复用4.2A的已审核结果。", "",
             "| 组别 | Extraction Recall | Extraction Precision | Hallucination | Overselect | E2E TP/FP/FN | E2E F1 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name in ("baseline", "prompt_lookup"):
        q = scores[name]["extraction"]
        e = scores[name]["end_to_end"]
        lines.append(f"| {name} | {q['recall']:.2%} | {q['precision']:.2%} | "
                     f"{q['hallucinated_events']}/{q['emitted_events']} | "
                     f"{q['overselected_events']}/{q['emitted_events']} | "
                     f"{e['tp']}/{e['fp']}/{e['fn']} | {e['f1']:.2%} |")
    report = directory / "quality_reviewed.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("QUALITY=" + str(report), flush=True)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark 4.2B prompt-lookup speculative decoding")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "smoke", "run", "summary", "run-downstream"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--baseline-result", type=Path, default=BASELINE_RESULT)
        if name in ("smoke", "run"):
            cmd.add_argument("--prompt-lookup-num-tokens", type=int, default=5)
            cmd.add_argument("--device", choices=("cuda",), default="cuda")
        if name == "run":
            cmd.add_argument("--output", type=Path)
        if name in ("summary", "run-downstream"):
            cmd.add_argument("--result-dir", type=Path, required=True)
        if name == "run-downstream":
            cmd.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            inputs = validate_b(args.baseline_result.resolve())
            print(json.dumps({"stories": len(inputs["cases"]), "baseline": str(args.baseline_result),
                              "experiment": "4.2B", "batch_size": 1}, ensure_ascii=False, indent=2))
            return 0
        if args.command == "smoke":
            validate_b(args.baseline_result.resolve())
            return smoke(args)
        if args.command == "run":
            return run(args)
        if args.command == "run-downstream":
            return run_downstream(args)
        baseline = args.baseline_result.resolve()
        inputs = validate_b(baseline)
        directory = args.result_dir.resolve()
        return 0 if summarize(directory, inputs, baseline, _read(directory / "manifest.json")["prompt_lookup_num_tokens"]) else 2
    except (ValueError, OSError, RuntimeError, KeyError, ImportError) as exc:
        parser.exit(2, f"错误：{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
