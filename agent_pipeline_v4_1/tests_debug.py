import unittest

from agent_pipeline_v4_1.debug_experiment import first_divergence, make_scenarios


class DebugExperimentTests(unittest.TestCase):
    def test_four_scenarios_isolate_neighbour_and_slot(self):
        specs = make_scenarios("T", ("N1", "N2", "N3"), ("M1", "M2", "M3"))
        self.assertEqual(specs, {
            "A": ["T"],
            "B": ["N1", "N2", "N3", "T"],
            "C": ["M1", "M2", "M3", "T"],
            "D": ["T", "N1", "N2", "N3"],
        })

    def test_rejects_duplicate_or_overlapping_neighbours(self):
        with self.assertRaises(ValueError):
            make_scenarios("T", ("N1", "N1", "N3"), ("M1", "M2", "M3"))
        with self.assertRaises(ValueError):
            make_scenarios("T", ("N1", "N2", "N3"), ("N1", "M2", "M3"))

    def test_first_divergence_preserves_logit_evidence(self):
        left = [{"token_id": 10, "top2": [{"token_id": 10, "logit": 2.0}, {"token_id": 11, "logit": 1.0}]},
                {"token_id": 12, "top2": [{"token_id": 12, "logit": 1.1}, {"token_id": 13, "logit": 1.0}]}]
        right = [{"token_id": 10, "top2": [{"token_id": 10, "logit": 2.0}, {"token_id": 11, "logit": 1.0}]},
                 {"token_id": 13, "top2": [{"token_id": 13, "logit": 1.2}, {"token_id": 12, "logit": 1.0}]}]
        result = first_divergence(left, right)
        self.assertEqual(result["first_divergence_step"], 2)
        self.assertEqual(result["left"]["top2"][0]["token_id"], 12)
        self.assertEqual(result["right"]["top2"][0]["token_id"], 13)

    def test_length_difference_is_divergence(self):
        result = first_divergence([{"token_id": 1}], [{"token_id": 1}, {"token_id": 2}])
        self.assertEqual(result["first_divergence_step"], 2)
        self.assertIsNone(result["left"])

    def test_tiny_qwen_probe_and_reordering_do_not_mutate_generation(self):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from unittest.mock import patch
        from agent_pipeline_v4_1.debug_experiment import run_scenario

        torch.manual_seed(7)
        config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                            num_hidden_layers=2, num_attention_heads=4,
                            num_key_value_heads=2, head_dim=8,
                            max_position_embeddings=128, eos_token_id=63, pad_token_id=0)
        model = Qwen3ForCausalLM(config).eval()

        class Tokenizer:
            def apply_chat_template(self, messages, **_):
                return [int(x) for x in messages[0]["content"].split(",")]

            def decode(self, tokens, skip_special_tokens=True):
                return ",".join(str(x) for x in tokens if x != 63)

        adapter = type("Adapter", (), {"model": model, "tokenizer": Tokenizer(),
                                        "torch": torch, "device": "cpu", "load_seconds": 0.0})()
        prompts = {"T": "4,5,6", "N1": "7,8", "N2": "9,10,11", "N3": "12,13"}
        requests = {name: {"messages": [{"role": "user", "content": prompt}],
                           "prompt_sha256": name, "spans": {"S1": {"text": "x"}},
                           "window": {"target_ids": ["S1"], "context_ids": []}}
                    for name, prompt in prompts.items()}
        with patch("agent_pipeline_v4_1.debug_experiment.MAX_NEW_TOKENS", 3):
            alone = run_scenario(adapter, "T", ["T"], requests)
            mixed = run_scenario(adapter, "T", ["N1", "N2", "N3", "T"], requests)
            moved = run_scenario(adapter, "T", ["T", "N1", "N2", "N3"], requests)
        self.assertEqual(alone["target_tokens"], mixed["target_tokens"])
        self.assertEqual(alone["target_tokens"], moved["target_tokens"])
        self.assertEqual(len(mixed["target_trace"]), len(mixed["target_tokens"]))
        self.assertEqual(mixed["target_initial_slot"], 3)


if __name__ == "__main__":
    unittest.main()
