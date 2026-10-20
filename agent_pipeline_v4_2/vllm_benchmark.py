"""4.2C isolated vLLM batching/ngram matrix; GPU measurements are user-run."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

from agent_pipeline_v2_2.runner import atomic_write_json, write_jsonl_atomic
from agent_pipeline_v4_1.audit import event_signature, score_extraction, validate_labels
from . import vllm_model as backend

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / ".cache/benchmark_4_2C_nf4"
SCREEN_IDS = ("SL-P03", "SL-P04", "SL-010", "SL-017")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


def make_arms(batches, ngrams):
    if (not batches or not ngrams or len(set(batches)) != len(batches)
            or len(set(ngrams)) != len(ngrams) or 0 not in ngrams
            or any(b not in (1, 2, 4) for b in batches) or any(n not in (0, 3, 5) for n in ngrams)):
        raise ValueError("unique batches 1/2/4 and ngrams 0/3/5 required, including baseline 0")
    return [{"name": f"batch_{b}_ngram_{n}", "batch_size": b, "ngram": n}
            for b in batches for n in ngrams]


def ensure_manifest(directory, identity):
    path = directory / "manifest.json"
    if path.exists():
        if read(path) != identity:
            raise ValueError("4.2C manifest changed; use a new output directory")
    else:
        if directory.exists() and any(directory.iterdir()):
            raise ValueError("output directory is nonempty without a manifest")
        atomic_write_json(path, identity)


def source_identity():
    from agent_pipeline_v4.cli import validate_inputs
    inputs = validate_inputs()
    identity = inputs["identity"]
    for filename in ("vllm_model.py", "vllm_benchmark.py", "quality.py"):
        identity["files"]["agent_pipeline_v4_2/" + filename] = backend.sha(ROOT / "agent_pipeline_v4_2" / filename)
    return inputs, json.loads(json.dumps(identity))


def fresh_review(rows, digest):
    cases, pending = {}, []
    for row in rows:
        cid = row["case_id"]
        cases[cid] = {}
        for event in row["result"]["benchmark_stages"]["extraction"]["events"]:
            cases[cid][event["id"]] = {"signature": event_signature(event), "label": None,
                                       "gold_ids": [], "note": None}
            pending.append(cid + "/" + event["id"])
    return {"schema_version": "benchmark-4.2C-review-v1", "extraction_sha256": digest,
            "cases": cases, "pending": pending,
            "provenance": "fresh-review-no-historical-label-reuse"}


def summarize(directory):
    manifest = read(directory / "manifest.json")
    results = {}
    lines = ["# Benchmark 4.2C: vLLM batching + n-gram", "",
             f"Phase: {manifest['phase']}. Same-backend comparison; not equivalent to historic HF scores.",
             "Bounded batch dispatch, not cross-batch continuous admission. Each arm has a fresh engine.",
             "Wall excludes model loading and warmup. Resumed arms have no comparable wall time.",
             "TTFT, prefill/decode split, speculative acceptance and worker GPU memory are not measured; never reported as zero.", "",
             "| Arm | Status | Wall s | Output tokens | Actual average batch | Recall | Precision | Pending |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for arm in manifest["arms"]:
        name = arm["name"]
        path = directory / name / "result.json"
        result = read(path) if path.exists() else {"status": "not_run"}
        results[name] = result
        quality = result.get("quality") or {}
        values = [name, result["status"], result.get("wall_seconds"), result.get("output_tokens"),
                  result.get("actual_average_batch"), quality.get("recall"), quality.get("precision"),
                  result.get("pending_count")]
        lines.append("| " + " | ".join("—" if v is None else f"{v:.4f}" if isinstance(v, float) else str(v)
                                       for v in values) + " |")
    report = {"phase": manifest["phase"], "arms": results}
    atomic_write_json(directory / "benchmark_4_2C_summary.json", report)
    target = directory / "benchmark_4_2C_summary.md"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("SUMMARY=" + str(target), flush=True)
    return report


def worker(args):
    from agent_pipeline_v4.static import run_static_cases
    directory = args.result_dir.resolve()
    manifest = read(directory / "manifest.json")
    inputs, source = source_identity()
    if source != manifest["source"] or backend.preflight() != manifest["runtime"]:
        raise ValueError("source/runtime changed since matrix started")
    model_dir = Path(manifest["model_dir"])
    if backend.validate_artifact(model_dir) != manifest["artifact"]:
        raise ValueError("model export changed since matrix started")
    arm = next(a for a in manifest["arms"] if a["name"] == args.arm)
    output = directory / args.arm
    output.mkdir(exist_ok=True)
    previous = jsonl(output / "extractor_runs.jsonl")
    if (output / "review.json").exists():
        raise ValueError("arm already has a review ledger; use a new result directory to rerun")
    # Persist interruption marker before loading, so any resumed partial session loses wall comparability.
    resumed = (output / "started.json").exists()
    atomic_write_json(output / "started.json", {"resumed": resumed})
    try:
        model = backend.VllmExtractorModel(model_dir, arm["batch_size"], arm["ngram"], manifest["gpu_memory"])
        cases = [c for c in inputs["cases"] if c["id"] in manifest["case_ids"]]
        measured = run_static_cases(cases, model, arm["batch_size"], output, warmup=True,
                                    progress=lambda done, total, cid: print(f"[{args.arm}] {done}/{total} {cid}", flush=True))
        rows, calls = measured["rows"], measured["calls"]
        complete = len(rows) == len(cases) and all(r["status"] == "ok" for r in rows)
        review_path = output / "review.json"
        review = fresh_review(rows, backend.sha(output / "extractor_runs.jsonl"))
        atomic_write_json(review_path, review)
        result = {"status": "ok" if complete else "partial", "quality": None,
                  "wall_seconds": None if resumed or previous else measured["wall_seconds"],
                  "resumed": resumed, "model_load_seconds": model.load_seconds,
                  "output_tokens": sum(c["output_tokens"] for c in calls),
                  "input_tokens": sum(c["input_tokens"] for c in calls), "generation_batches": len(calls),
                  "actual_average_batch": sum(c["batch_size"] for c in calls) / len(calls) if calls else None,
                  "pending_count": len(review["pending"]),
                  "extraction_sha256": backend.sha(output / "extractor_runs.jsonl"),
                  "calls_sha256": backend.sha(output / "inference_calls.jsonl"),
                  "status_counts": {s: sum(r["status"] == s for r in rows) for s in ("ok", "partial", "error")}}
        atomic_write_json(output / "result.json", result)
        return 0 if complete else 2
    except Exception as exc:
        atomic_write_json(output / "result.json", {"status": "error", "error": f"{type(exc).__name__}: {exc}",
                                                   "quality": None, "wall_seconds": None})
        raise


def run_matrix(args):
    runtime = backend.preflight()
    artifact = backend.validate_artifact(args.model_dir)
    inputs, source = source_identity()
    arms = make_arms(args.batches, args.ngrams)
    cases = inputs["cases"] if args.phase == "full" else [c for c in inputs["cases"] if c["id"] in SCREEN_IDS]
    if args.phase == "screen" and len(cases) != len(SCREEN_IDS):
        raise ValueError("screen dataset missing expected cases")
    backend.engine_options(args.model_dir, args.batches[0], args.ngrams[0], args.gpu_memory)
    directory = args.output.resolve() if args.output else ROOT / "agent_pipeline_v4_2/reports" / (
        "benchmark_4_2C_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6])
    identity = {"schema_version": "benchmark-4.2C-v1", "source": source, "runtime": runtime,
                "artifact": artifact, "model_dir": str(args.model_dir.resolve()), "phase": args.phase,
                "case_ids": [c["id"] for c in cases], "arms": arms, "gpu_memory": args.gpu_memory,
                "batch_dispatch": "static bounded groups; independent requests, no concatenation"}
    ensure_manifest(directory, identity)
    print("RESULT_DIR=" + str(directory), flush=True)
    failed = False
    for arm in arms:
        output = directory / arm["name"]
        result_path = output / "result.json"
        if result_path.exists() and read(result_path).get("status") in ("ok", "partial"):
            saved = read(result_path)
            if (saved["extraction_sha256"] != backend.sha(output / "extractor_runs.jsonl")
                    or saved["calls_sha256"] != backend.sha(output / "inference_calls.jsonl")):
                raise ValueError("saved arm data changed")
            failed |= saved["status"] != "ok"
            continue
        output.mkdir(exist_ok=True)
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", VLLM_NO_USAGE_STATS="1",
                   TOKENIZERS_PARALLELISM="false", VLLM_WORKER_MULTIPROC_METHOD="spawn")
        print("START=" + arm["name"], flush=True)
        with (output / "worker.log").open("a", encoding="utf-8") as log:
            process = subprocess.Popen([sys.executable, "-u", "-m", "agent_pipeline_v4_2.vllm_benchmark",
                                        "_worker", "--result-dir", str(directory), "--arm", arm["name"]],
                                       cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace")
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
            except KeyboardInterrupt:
                process.terminate()
                process.wait()
                raise
        if code:
            failed = True
            if not result_path.exists() or read(result_path).get("status") == "ok":
                atomic_write_json(result_path, {"status": "error", "exit_code": code, "quality": None})
        summarize(directory)
        # A failed engine load is unlikely to improve in a nine-arm sweep. Stop for diagnosis.
        if code and read(result_path).get("status") == "error":
            return 2
    summarize(directory)
    return 2 if failed else 0


def check_review(directory, arm, cases):
    output = directory / arm
    rows = jsonl(output / "extractor_runs.jsonl")
    review = read(output / "review.json")
    result = read(output / "result.json")
    if result.get("status") != "ok" or any(row.get("status") != "ok" for row in rows):
        raise ValueError(f"{arm}: incomplete extraction cannot be quality-approved")
    digest = backend.sha(output / "extractor_runs.jsonl")
    if review.get("pending") or review.get("extraction_sha256") != digest or result.get("extraction_sha256") != digest:
        raise ValueError(f"{arm}: pending review or extraction changed")
    if result.get("calls_sha256") != backend.sha(output / "inference_calls.jsonl"):
        raise ValueError(f"{arm}: inference traces changed")
    validate_labels(cases, rows, review["cases"])
    return rows, review, result


def score(args):
    directory = args.result_dir.resolve()
    manifest = read(directory / "manifest.json")
    inputs, identity = source_identity()
    if manifest["phase"] != "full" or identity != manifest["source"]:
        raise ValueError("quality requires a full run with unchanged source inputs")
    for arm in manifest["arms"]:
        name = arm["name"]
        rows, review, result = check_review(directory, name, inputs["cases"])
        result["quality"] = score_extraction(inputs["cases"], rows, review["cases"])
        result["pending_count"] = 0
        atomic_write_json(directory / name / "result.json", result)
    summarize(directory)
    return 0


def downstream_score(cases, rows, runs, review):
    from .quality import score_reviewed
    result = score_reviewed(cases, rows, runs, review)
    result["status_counts"] = {status: sum(r["status"] == status for r in runs)
                               for status in ("ok", "partial", "error")}
    result["complete"] = len(runs) == len(cases) and all(r["status"] == "ok" for r in runs)
    return result


def downstream(args):
    # Run this in the original Windows environment, preserving frozen Retrieval/Judge.
    directory = args.result_dir.resolve()
    manifest = read(directory / "manifest.json")
    inputs, identity = source_identity()
    if manifest["phase"] != "full" or identity != manifest["source"]:
        raise ValueError("downstream requires full run and unchanged source")
    arms = args.arms or [a["name"] for a in manifest["arms"]]
    if not arms or len(set(arms)) != len(arms) or not set(arms) <= {a["name"] for a in manifest["arms"]}:
        raise ValueError("unknown or duplicate downstream arm")
    checked = {name: check_review(directory, name, inputs["cases"]) for name in arms}
    from agent_pipeline_v3.model import ProfiledV2QwenJudge
    from agent_pipeline_v2_1.retrieval import V21Retriever
    from agent_pipeline_v4_1.cli import _run_one
    from encoder import Encoder
    from retrieval import Retriever
    # Bind downstream resume to source/index and reviewed input as well as runtime versions.
    import importlib.metadata
    versions = {p: importlib.metadata.version(p) for p in ("torch", "transformers", "bitsandbytes")}
    index_hashes = {str(p.relative_to(ROOT)): backend.sha(p) for p in sorted((ROOT / "data/index").rglob("*")) if p.is_file()}
    downstream_code = {p.relative_to(ROOT).as_posix(): backend.sha(p)
                       for folder in ("agent_pipeline_v2", "agent_pipeline_v2_1", "agent_pipeline_v4_1")
                       for p in sorted((ROOT / folder).rglob("*")) if p.is_file() and p.suffix in (".py", ".txt")
                       and "reports" not in p.relative_to(ROOT).parts}
    downstream_code.update({name: backend.sha(ROOT / name)
                            for name in ("encoder.py", "retrieval.py", "index_store.py")})
    for name in arms:
        path = directory / name / "downstream"
        ensure_manifest(path, {"versions": versions, "index": index_hashes, "code": downstream_code,
                               "review": backend.sha(directory / name / "review.json"),
                               "extraction": checked[name][2]["extraction_sha256"],
                               "metadata": backend.sha(ROOT / "evaluation/agent_pipeline_v2_1_lore_metadata.json")})
    llm = ProfiledV2QwenJudge("cuda")
    dense = Retriever(ROOT / "data/index", Encoder(device=llm.device, offline=True, precision="float32"))
    retriever = V21Retriever(dense, read(ROOT / "evaluation/agent_pipeline_v2_1_lore_metadata.json"))
    scores = read(directory / "quality_reviewed.json") if (directory / "quality_reviewed.json").exists() else {}
    for name in arms:
        rows, review, result = checked[name]
        path = directory / name / "downstream/downstream_runs.jsonl"
        runs = jsonl(path)
        seen = [r["case_id"] for r in runs]
        if len(seen) != len(set(seen)) or not set(seen) <= set(manifest["case_ids"]):
            raise ValueError("duplicate/unknown downstream rows")
        by_id = {r["case_id"]: r for r in rows}
        for case in inputs["cases"]:
            cid = case["id"]
            if cid in seen:
                continue
            llm.set_story_id(name + ":" + cid)
            extraction = by_id[cid]["result"]["benchmark_stages"]["extraction"]
            runs.append({"case_id": cid, "system": name, **_run_one(extraction, llm, retriever)})
            write_jsonl_atomic(path, runs)
            llm.clear_traces()
            print(f"[{name} downstream] {len(runs)}/24 {cid}", flush=True)
        scores[name] = downstream_score(inputs["cases"], rows, runs, review)
        atomic_write_json(directory / "quality_reviewed.json", scores)
    lines = ["# 4.2C reviewed downstream", "", "| Arm | Complete | TP | FP | FN | F1 |", "|---|---|---:|---:|---:|---:|"]
    for name, value in scores.items():
        e = value["end_to_end"]
        f1 = "—" if e['f1'] is None else f"{e['f1']:.2%}"
        lines.append(f"| {name} | {value['complete']} | {e['tp']} | {e['fp']} | {e['fn']} | {f1} |")
    target = directory / "quality_reviewed.md"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("QUALITY=" + str(target), flush=True)
    return 0 if all(scores[name]["complete"] for name in arms) else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("export-model", "preflight", "run"):
        p = sub.add_parser(name)
        p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
        if name == "run":
            p.add_argument("--phase", choices=("screen", "full"), default="screen")
            p.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4])
            p.add_argument("--ngrams", type=int, nargs="+", default=[0, 3, 5])
            p.add_argument("--gpu-memory", type=float, default=.80)
            p.add_argument("--output", type=Path)
    for name in ("summary", "score", "run-downstream", "_worker"):
        p = sub.add_parser(name)
        p.add_argument("--result-dir", type=Path, required=True)
        if name == "_worker":
            p.add_argument("--arm", required=True)
        if name == "run-downstream":
            p.add_argument("--arms", nargs="+")
    args = parser.parse_args(argv)
    try:
        if args.command == "export-model":
            backend.export_model(args.model_dir.resolve())
            return 0
        if args.command == "preflight":
            runtime = backend.preflight()
            artifact = backend.validate_artifact(args.model_dir)
            inputs, _ = source_identity()
            print(json.dumps({"runtime": runtime, "model": artifact["model"], "stories": len(inputs["cases"]),
                              "notice": "environment checks only; engine compatibility needs screen run"}, indent=2))
            return 0
        if args.command == "run":
            return run_matrix(args)
        if args.command == "_worker":
            return worker(args)
        if args.command == "summary":
            summarize(args.result_dir)
            return 0
        return score(args) if args.command == "score" else downstream(args)
    except (ValueError, OSError, RuntimeError, KeyError, ImportError, StopIteration) as exc:
        parser.exit(2, f"错误：{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
