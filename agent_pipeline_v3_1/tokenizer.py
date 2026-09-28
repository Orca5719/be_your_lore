from __future__ import annotations

from pathlib import Path

from qwen_judge import MODEL, REVISION


ROOT = Path(__file__).resolve().parents[1]


def load_token_counter():
    """Load only the pinned tokenizer. No model weights or generation code are used."""
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    snapshot = snapshot_download(
        MODEL,
        revision=REVISION,
        cache_dir=ROOT / ".cache" / "huggingface",
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    return count
