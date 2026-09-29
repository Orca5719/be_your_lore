from agent_pipeline_v3.quality import build_quality_reference, compare_quality


def row(verdict="contradiction", seconds=1.0, dense_score=0.75, evidence_id="L1"):
    return {
        "case_id": "SL-001",
        "status": "ok",
        "stage_metrics": {"total": {"seconds": seconds}},
        "result": {
            "findings": [{"id": "R1", "event_ids": ["E1"], "verdict": verdict, "reasons": ["x"], "citations": []}],
            "benchmark_stages": {
                "extraction": {
                    "status": "ok",
                    "events": [{"id": "E1", "actors": ["雷"], "event": "x", "source_ids": ["S1"]}],
                },
                "judge": {
                    "status": "ok",
                    "items": [{
                        "event_id": "E1",
                        "status": "ok",
                        "verdict": verdict,
                        "origin": "model",
                        "reason": "x",
                        "evidence": [{"id": evidence_id, "text": "lore", "rank": 1, "dense_score": dense_score}],
                        "citations": [{"evidence_id": evidence_id, "chunk_id": evidence_id, "quote": "lore"}],
                    }],
                },
            },
        },
    }


def test_quality_signature_ignores_timing_but_detects_semantic_change():
    reference = build_quality_reference([row(seconds=1.0)])
    same = compare_quality([row(seconds=999.0)], reference)
    changed = compare_quality([row(verdict="uncertain")], reference)
    assert same["comparable"] is True
    assert same["matched_cases"] == 1
    assert changed["comparable"] is False
    assert changed["mismatched_case_ids"] == ["SL-001"]


def test_quality_signature_ignores_retrieval_float_noise_but_detects_changed_evidence():
    reference = build_quality_reference([row(dense_score=0.75000000000001)])

    float_noise = compare_quality([row(dense_score=0.75000000000002)], reference)
    changed_evidence = compare_quality([row(evidence_id="L2")], reference)

    assert float_noise["comparable"] is True
    assert changed_evidence["comparable"] is False
