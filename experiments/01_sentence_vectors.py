"""实验 01：观察三句话如何变成向量；示例不代表正式 canon。"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
parser.add_argument("--offline", action="store_true")
args = parser.parse_args()
device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
if device == "auto":
    device = "cpu"
if device == "cuda" and not torch.cuda.is_available():
    parser.error("CUDA 不可用，请使用 --device cpu")

root = Path(__file__).resolve().parents[1]
cache = root / ".cache" / "huggingface"
model_id = "BAAI/bge-small-zh-v1.5"
texts = [
    "雷的左心脏寄宿着亚巴顿。",
    "亚巴顿栖居在雷胸腔左侧的心脏里。",
    "德尔塔穿着一套动力外骨骼。",
]
# 本实验比较句子相似度，三句均用正文编码，不添加检索提示。
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache, local_files_only=args.offline)
model = AutoModel.from_pretrained(model_id, cache_dir=cache, local_files_only=args.offline, use_safetensors=True)
model = model.to(device).float().eval()
inputs = tokenizer(texts, padding=True, truncation=False, return_tensors="pt")
if inputs["input_ids"].shape[1] > 512:
    raise ValueError("输入超过 512 tokens；实验禁止静默截断")

print("测试设定，不代表正式 canon")
print("设备:", device, "参数类型:", next(model.parameters()).dtype)
for i, text in enumerate(texts):
    print(f"\n句子 {i}: {text}")
    print("tokens:", tokenizer.convert_ids_to_tokens(inputs["input_ids"][i].tolist()))
    print("input_ids:", inputs["input_ids"][i].tolist())
    print("attention_mask:", inputs["attention_mask"][i].tolist())
print("\ninput_ids shape:", tuple(inputs["input_ids"].shape))
inputs = {key: value.to(device) for key, value in inputs.items()}
with torch.inference_mode():
    hidden = model(**inputs).last_hidden_state
    cls = hidden[:, 0, :]
    vectors = F.normalize(cls, p=2, dim=1)
    norms = vectors.norm(dim=1)
    similarities = vectors @ vectors.T
    assert torch.isfinite(vectors).all(), "向量包含非有限数值"
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "归一化失败"
    assert torch.allclose(similarities.diag(), torch.ones_like(norms), atol=1e-5)
    cosine = F.cosine_similarity(cls[:, None, :], cls[None, :, :], dim=-1)
    assert torch.allclose(similarities, cosine, atol=1e-5), "点积与余弦不一致"

print("last_hidden_state shape:", tuple(hidden.shape))
print("CLS shape:", tuple(cls.shape))
print("CLS 前 8 维:", cls[0, :8].cpu().tolist())
print("归一化向量前 8 维:", vectors[0, :8].cpu().tolist())
print("归一化后长度:", norms.cpu().tolist())
print("相似度矩阵（行列依次对应三句话）:\n", similarities.cpu().numpy())
print("相似度不是事实正确率，也不是矛盾置信度。")
report = {
    "model": model_id,
    "revision": getattr(model.config, "_commit_hash", None),
    "device": device,
    "dtype": "float32",
    "torch": torch.__version__,
    "texts": texts,
    "input_shape": list(inputs["input_ids"].shape),
    "hidden_shape": list(hidden.shape),
    "vector_shape": list(vectors.shape),
    "norms": norms.cpu().tolist(),
    "similarities": similarities.cpu().tolist(),
    "checks": "finite vectors, unit norms, self similarity, dot equals cosine: passed",
}
report_dir = root / "reports"
report_dir.mkdir(exist_ok=True)
np.save(report_dir / "experiment01_vectors.npy", vectors.cpu().numpy())
(report_dir / "experiment01.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
