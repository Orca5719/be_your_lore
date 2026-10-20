import json
from types import SimpleNamespace as NS

import pytest

from agent_pipeline_v4_2 import vllm_model as backend
from agent_pipeline_v4_2 import vllm_benchmark as bench


def test_matrix_has_paired_baselines_and_rejects_duplicate_arms():
    arms = bench.make_arms([1, 2, 4], [0, 3, 5])
    assert len(arms) == 9
    assert arms[0] == {"name": "batch_1_ngram_0", "batch_size": 1, "ngram": 0}
    with pytest.raises(ValueError):
        bench.make_arms([1, 1], [0, 3])
    with pytest.raises(ValueError):
        bench.make_arms([1], [3])


def test_engine_options_keep_prequantized_weights_and_disable_prefix_cache(tmp_path):
    opts = backend.engine_options(tmp_path, 4, 3, .8)
    assert opts["max_num_seqs"] == 4
    assert opts["quantization"] == "bitsandbytes"
    assert opts["enable_prefix_caching"] is False
    assert opts["async_scheduling"] is False
    assert backend.engine_options(tmp_path, 1, 0, .8)["async_scheduling"] is False
    assert opts["speculative_config"] == {"method": "ngram", "num_speculative_tokens": 3,
                                           "prompt_lookup_min": 2, "prompt_lookup_max": 5}
    assert "speculative_config" not in backend.engine_options(tmp_path, 1, 0, .8)


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return list(range(int(messages[0]["content"])))


class Engine:
    def generate(self, prompts, params, use_tqdm):
        return [NS(prompt_token_ids=p["prompt_token_ids"], metrics=None,
                   outputs=[NS(text=str(len(p["prompt_token_ids"])), token_ids=[7, 8],
                               finish_reason="length" if i else "stop")])
                for i, p in enumerate(prompts)]


def test_adapter_preserves_row_mapping_and_marks_length_as_truncated():
    model = backend.VllmExtractorModel.__new__(backend.VllmExtractorModel)
    model.tokenizer, model.engine = Tokenizer(), Engine()
    model.sampling_factory = lambda **kwargs: kwargs
    model.batch_size = 2
    model.sequence = 0
    model.eos_ids = [151645, 151643]
    sampled = []
    model.sampling_factory = lambda **kwargs: sampled.append(kwargs) or kwargs
    result = model._generate_batch([[{"content": "3"}], [{"content": "5"}]], 10)
    assert result == ["3", "5"]
    timing = model.last_batch_generation
    assert timing["input_tokens"] == [3, 5]
    assert timing["truncated_indices"] == [1]
    assert timing["profile"]["ttft_ms"] is None
    assert timing["profile"]["decode_time"] is None
    assert sampled[0]["stop_token_ids"] == [151645, 151643]
    with pytest.raises(ValueError, match="4096"):
        model._generate_batch([[{"content": "4090"}]], 10)


def test_artifact_validation_detects_weight_changes(tmp_path):
    from qwen_judge import MODEL, REVISION
    (tmp_path / "config.json").write_text(json.dumps({"quantization_config": {
        "load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": "bfloat16"}}))
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "tokenizer_config.json").write_text("{}")
    manifest = {"model": MODEL, "revision": REVISION, "files": backend.file_hashes(tmp_path)}
    (tmp_path / "worldcheck_export.json").write_text(json.dumps(manifest))
    assert backend.validate_artifact(tmp_path)["model"] == MODEL
    (tmp_path / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact"):
        backend.validate_artifact(tmp_path)


def test_resume_rejects_changed_manifest_without_overwrite(tmp_path):
    bench.ensure_manifest(tmp_path, {"ngram": 3})
    with pytest.raises(ValueError):
        bench.ensure_manifest(tmp_path, {"ngram": 5})
    assert json.loads((tmp_path / "manifest.json").read_text()) == {"ngram": 3}


def test_fresh_review_never_inherits_old_labels():
    event = {"id": "E1", "event": "test"}
    rows = [{"case_id": "A", "status": "ok", "result": {"benchmark_stages": {
        "extraction": {"events": [event]}}}}]
    review = bench.fresh_review(rows, "abc")
    assert review["pending"] == ["A/E1"]
    assert review["cases"]["A"]["E1"]["label"] is None


def test_summary_keeps_missing_metrics_unknown_and_screen_not_quality(tmp_path):
    bench.ensure_manifest(tmp_path, {"phase": "screen", "case_ids": ["A"],
                                    "arms": [{"name": "batch_1_ngram_0"}]})
    arm = tmp_path / "batch_1_ngram_0"
    arm.mkdir()
    (arm / "result.json").write_text(json.dumps({"status": "ok", "wall_seconds": 2,
                                               "quality": None, "output_tokens": 10}))
    report = bench.summarize(tmp_path)
    assert report["phase"] == "screen"
    assert report["arms"]["batch_1_ngram_0"]["quality"] is None


def test_source_identity_survives_json_roundtrip():
    _, identity = bench.source_identity()
    assert json.loads(json.dumps(identity)) == identity


def test_screen_refuses_quality_scoring(tmp_path, monkeypatch):
    bench.ensure_manifest(tmp_path, {"phase": "screen", "source": {}})
    monkeypatch.setattr(bench, "source_identity", lambda: ({"cases": []}, {}))
    with pytest.raises(ValueError, match="full"):
        bench.score(NS(result_dir=tmp_path))


def test_vllm_adapter_runs_real_extractor_queue(tmp_path):
    from agent_pipeline_v4.static import run_static_cases
    class TextTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return [ord(c) for c in messages[1]["content"]]
    class ExtractEngine:
        def generate(self, prompts, params, use_tqdm):
            results = []
            for p in prompts:
                payload = json.loads("".join(chr(c) for c in p["prompt_token_ids"]))
                raw = json.dumps({"events": [], "ignored_spans": [],
                                  "non_event_span_ids": list(payload["target_spans"])})
                results.append(NS(prompt_token_ids=p["prompt_token_ids"], metrics=None,
                                  outputs=[NS(text=raw, token_ids=[1, 2], finish_reason="stop")]))
            return results
    model = backend.VllmExtractorModel.__new__(backend.VllmExtractorModel)
    model.tokenizer, model.engine = TextTokenizer(), ExtractEngine()
    model.sampling_factory = lambda **kw: kw
    model.batch_size, model.sequence, model.load_seconds, model.eos_ids = 2, 0, 0, [151645, 151643]
    cases = [{"id": "A", "story": "雷走入房间。"}, {"id": "B", "story": "月城抬起手。"}]
    result = run_static_cases(cases, model, 2, tmp_path, warmup=False)
    assert [r["case_id"] for r in result["rows"]] == ["A", "B"]
    assert all(r["status"] == "ok" for r in result["rows"])
    assert sum(c["batch_size"] for c in result["calls"]) == 2
    assert (tmp_path / "extractor_runs.jsonl").is_file()


def test_review_rejects_modified_extraction_even_when_pending_cleared(tmp_path):
    arm = tmp_path / "arm"
    arm.mkdir()
    rows = [{"case_id": "A", "status": "ok", "result": {"benchmark_stages": {
        "extraction": {"events": []}}}}]
    bench.write_jsonl_atomic(arm / "extractor_runs.jsonl", rows)
    bench.write_jsonl_atomic(arm / "inference_calls.jsonl", [])
    digest = backend.sha(arm / "extractor_runs.jsonl")
    bench.atomic_write_json(arm / "review.json", bench.fresh_review(rows, digest))
    bench.atomic_write_json(arm / "result.json", {"status": "ok", "extraction_sha256": digest,
                                                "calls_sha256": backend.sha(arm / "inference_calls.jsonl")})
    cases = [{"id": "A", "gold_facts": []}]
    assert bench.check_review(tmp_path, "arm", cases)[0] == rows
    result = bench.read(arm / "result.json")
    result["status"] = "partial"
    bench.atomic_write_json(arm / "result.json", result)
    with pytest.raises(ValueError, match="incomplete"):
        bench.check_review(tmp_path, "arm", cases)
    result["status"] = "ok"
    bench.atomic_write_json(arm / "result.json", result)
    (arm / "extractor_runs.jsonl").write_text("\n")
    with pytest.raises(ValueError, match="changed"):
        bench.check_review(tmp_path, "arm", cases)


def test_worker_marks_resumed_wall_noncomparable(tmp_path, monkeypatch):
    from agent_pipeline_v4 import static
    source = {"test": True}
    cases = [{"id": "A", "story": "雷走入房间。"}]
    arm = {"name": "batch_1_ngram_0", "batch_size": 1, "ngram": 0}
    bench.ensure_manifest(tmp_path, {"source": source, "runtime": {}, "artifact": {},
                                    "model_dir": str(tmp_path), "arms": [arm], "gpu_memory": .8,
                                    "phase": "screen", "case_ids": ["A"]})
    output = tmp_path / arm["name"]
    output.mkdir()
    bench.atomic_write_json(output / "started.json", {})
    monkeypatch.setattr(bench, "source_identity", lambda: ({"cases": cases}, source))
    monkeypatch.setattr(backend, "preflight", lambda: {})
    monkeypatch.setattr(backend, "validate_artifact", lambda path: {})
    monkeypatch.setattr(backend, "VllmExtractorModel", lambda *args: NS(load_seconds=1))
    def run(cases, model, size, output, **kwargs):
        rows = [{"case_id": "A", "status": "ok", "result": {"benchmark_stages": {
            "extraction": {"events": []}}}}]
        bench.write_jsonl_atomic(output / "extractor_runs.jsonl", rows)
        bench.write_jsonl_atomic(output / "inference_calls.jsonl", [])
        return {"rows": rows, "calls": [], "wall_seconds": 2}
    monkeypatch.setattr(static, "run_static_cases", run)
    assert bench.worker(NS(result_dir=tmp_path, arm=arm["name"])) == 0
    assert bench.read(output / "result.json")["wall_seconds"] is None
    assert bench.read(output / "result.json")["quality"] is None


def test_downstream_score_exposes_failure_even_with_perfect_f1(monkeypatch):
    from agent_pipeline_v4_2 import quality
    monkeypatch.setattr(quality, "score_reviewed", lambda *args: {"end_to_end": {"f1": 1.0}})
    result = bench.downstream_score([], [], [{"status": "error"}], {})
    assert result["complete"] is False
    assert result["status_counts"]["error"] == 1
