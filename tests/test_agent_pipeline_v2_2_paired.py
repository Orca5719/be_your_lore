from agent_pipeline_v2_2.paired import run_paired_matrix


def test_paired_reuses_events_and_retrieval_per_method():
    calls = []
    def retrieval(name):
        def run(extraction, top_k):
            calls.append((name, top_k))
            return {"method": name, "events": extraction["events"]}
        return run
    def judge(name):
        return lambda value, batch_size: {"judge": name, "method": value["method"], "items": []}
    result = run_paired_matrix([{"id": "E1", "event": "x"}], retrieval("dense"), retrieval("hybrid"), judge("v1"), judge("v2.1"))
    assert result["status"] == "ok"
    assert calls == [("dense", 5), ("hybrid", 5)]
    assert len({cell["event_digest"] for cell in result["cells"].values()}) == 1
    assert result["cells"]["dense_v21"]["result"] == {"judge": "v2.1", "method": "dense", "items": []}


def test_one_cell_failure_is_isolated():
    def judge_v1(value, batch_size):
        if value["method"] == "hybrid":
            raise RuntimeError("bad cell")
        return {"items": []}
    retrieve = lambda method: lambda extraction, top_k: {"method": method}
    result = run_paired_matrix([], retrieve("dense"), retrieve("hybrid"), judge_v1, lambda value, batch_size: {"items": []})
    assert result["status"] == "partial"
    assert result["cells"]["hybrid_v1"]["status"] == "error"
    assert sum(cell["status"] == "ok" for cell in result["cells"].values()) == 3
