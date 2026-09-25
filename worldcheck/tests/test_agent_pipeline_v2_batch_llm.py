import json
import unittest

import torch

from agent_pipeline_v2.batch_llm import call_json_batch, left_pad_rows
from agent_pipeline_v2.model import V2QwenJudge


class BatchLLMTransportTests(unittest.TestCase):
    def test_left_padding_preserves_row_tokens_and_attention_mask(self):
        batch = left_pad_rows([[11, 12, 13], [21]], pad_token_id=0, torch_module=torch, device="cpu")
        self.assertEqual(batch["input_ids"].tolist(), [[11, 12, 13], [0, 0, 21]])
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1, 1], [0, 0, 1]])
        self.assertEqual(batch["input_lengths"], [3, 1])
        self.assertEqual(batch["padded_input_tokens"], 2)

    def test_only_invalid_rows_enter_retry_batch(self):
        valid_a = json.dumps({"value": "A"})
        valid_b = json.dumps({"value": "B"})

        class LLM:
            def __init__(self):
                self.calls = []
                self.outputs = [[valid_a, "not json"], [valid_b]]
                self.last_batch_generation = {}

            def _generate_batch(self, messages, max_new_tokens):
                self.calls.append(messages)
                self.last_batch_generation = {
                    "seconds": 0.1,
                    "batch_size": len(messages),
                    "input_tokens": [10] * len(messages),
                    "padded_input_tokens": 0,
                    "generated_tokens": [4] * len(messages),
                }
                return self.outputs.pop(0)

        llm = LLM()
        requests = [
            {"request_id": "F1", "messages": [{"role": "user", "content": "a"}]},
            {"request_id": "F2", "messages": [{"role": "user", "content": "b"}]},
        ]
        validators = [lambda value: value["value"] == "A" or (_ for _ in ()).throw(ValueError("wrong A")),
                      lambda value: value["value"] == "B" or (_ for _ in ()).throw(ValueError("wrong B"))]
        values, report = call_json_batch(llm, requests, validators=validators, max_output=32)
        self.assertEqual(values, [{"value": "A"}, {"value": "B"}])
        self.assertEqual([len(call) for call in llm.calls], [2, 1])
        self.assertEqual(report["rows"][0]["attempts"], 1)
        self.assertEqual(report["rows"][1]["attempts"], 2)
        self.assertEqual(report["retry_count"], 1)
        self.assertIn("上次回复格式不合格", llm.calls[1][0][-1]["content"])

    def test_batch_runtime_error_marks_every_active_row(self):
        class LLM:
            last_batch_generation = {}

            def _generate_batch(self, messages, max_new_tokens):
                raise RuntimeError("CUDA out of memory")

        requests = [
            {"request_id": "F1", "messages": []},
            {"request_id": "F2", "messages": []},
        ]
        values, report = call_json_batch(LLM(), requests, validators=[None, None], max_output=32)
        self.assertEqual(values, [None, None])
        self.assertEqual(report["status"], "error")
        self.assertTrue(all("out of memory" in row["error"] for row in report["rows"]))

    def test_batch_budget_value_error_is_a_reported_failure(self):
        class LLM:
            last_batch_generation = {}

            def _generate_batch(self, messages, max_new_tokens):
                raise ValueError("批量输入和输出超过4096 tokens预算")

        values, report = call_json_batch(LLM(), [{"request_id": "F1", "messages": []}], validators=[None], max_output=32)
        self.assertEqual(values, [None])
        self.assertEqual(report["status"], "error")
        self.assertIn("4096", report["rows"][0]["error"])

    def test_request_and_validator_counts_must_match(self):
        with self.assertRaisesRegex(ValueError, "数量"):
            call_json_batch(object(), [{"request_id": "F1", "messages": []}], validators=[])

    def test_truncated_row_is_retried_even_if_prefix_is_valid_json(self):
        class LLM:
            def __init__(self):
                self.calls = 0
                self.last_batch_generation = {}

            def _generate_batch(self, messages, max_new_tokens):
                self.calls += 1
                self.last_batch_generation = {"seconds": 0.1, "batch_size": 1, "input_tokens": [10], "padded_input_tokens": 0, "generated_tokens": [4], "truncated_indices": [0] if self.calls == 1 else []}
                return ['{"value":"ok"}']

        llm = LLM()
        values, report = call_json_batch(llm, [{"request_id": "F1", "messages": []}], validators=[None], max_output=32)
        self.assertEqual(values, [{"value": "ok"}])
        self.assertEqual(report["rows"][0]["attempts"], 2)

    def test_qwen_adapter_generates_true_two_row_tensor_batch(self):
        class Tokenizer:
            pad_token_id = 0
            eos_token_id = 99

            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                return messages[0]["tokens"]

            def decode(self, ids, skip_special_tokens=True):
                return " ".join(str(token) for token in ids if token not in {0, 99})

        class Model:
            generation_config = type("Config", (), {"eos_token_id": 99})()

            def __init__(self):
                self.inputs = None

            def generate(self, **kwargs):
                self.inputs = kwargs
                suffix = torch.tensor([[31, 99], [41, 99]], dtype=torch.long)
                return torch.cat([kwargs["input_ids"], suffix], dim=1)

        judge = V2QwenJudge.__new__(V2QwenJudge)
        judge.device = "cpu"
        judge.torch = torch
        judge.tokenizer = Tokenizer()
        judge.model = Model()
        outputs = judge._generate_batch(
            [[{"tokens": [11, 12, 13]}], [{"tokens": [21]}]], max_new_tokens=2
        )
        self.assertEqual(outputs, ["31", "41"])
        self.assertEqual(judge.model.inputs["input_ids"].tolist(), [[11, 12, 13], [0, 0, 21]])
        self.assertEqual(judge.model.inputs["attention_mask"].tolist(), [[1, 1, 1], [0, 0, 1]])
        self.assertEqual(judge.last_batch_generation["input_tokens"], [3, 1])
        self.assertEqual(judge.last_batch_generation["padded_input_tokens"], 2)


if __name__ == "__main__":
    unittest.main()
