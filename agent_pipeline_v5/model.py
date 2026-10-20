"""Explicit shared cache loading; no downloads or persistent global changes."""
import importlib.metadata
import json
from pathlib import Path
from unittest.mock import patch

from qwen_judge import MODEL, REVISION


def versions():
    return {name:importlib.metadata.version(name) for name in ('torch','transformers','bitsandbytes','accelerate')}


def snapshot_path(asset_root):
    path = Path(asset_root) / '.cache/huggingface' / ('models--'+MODEL.replace('/','--')) / 'snapshots' / REVISION
    if not path.is_dir():
        raise ValueError('Pinned Qwen snapshot not found in shared cache; no download attempted: '+str(path))
    for name in ('config.json','generation_config.json','tokenizer_config.json'):
        if not (path/name).is_file():
            raise ValueError('incomplete local model snapshot: '+name)
    index = path / 'model.safetensors.index.json'
    shards = set(json.loads(index.read_text(encoding='utf-8'))['weight_map'].values()) if index.exists() else {'model.safetensors'}
    if any(not (path/name).is_file() for name in shards):
        raise ValueError('local model weight shards are missing')
    return path


def load_model(asset_root):
    snapshot_path(asset_root)
    if versions()['transformers'] != '4.57.6':
        raise ValueError('5B requires existing transformers==4.57.6; no installation attempted')
    import qwen_judge
    from agent_pipeline_v4_2.speculative_model import PromptLookupExtractorModel
    # Existing constructor resolves ROOT/.cache and uses local_files_only=True.
    # Only initialization is scoped; no concurrent model creation in this runner.
    with patch.object(qwen_judge, 'ROOT', Path(asset_root).resolve()):
        return PromptLookupExtractorModel('cuda', num_tokens=5)


def load_retriever(asset_root, device):
    """Same frozen BGE/Hybrid configuration with an explicit cache directory."""
    import encoder
    from transformers import AutoModel, AutoTokenizer
    from retrieval import Retriever
    from agent_pipeline_v2_1.retrieval import V21Retriever
    cache = Path(asset_root) / '.cache/huggingface'
    options = dict(cache_dir=cache, revision=encoder.REVISION, local_files_only=True)
    encoded = encoder.Encoder.__new__(encoder.Encoder)
    encoded.device = device
    encoded.tokenizer = AutoTokenizer.from_pretrained(encoder.MODEL, **options)
    encoded.model = AutoModel.from_pretrained(encoder.MODEL, use_safetensors=True, **options).to(device=device, dtype=encoder.torch.float32).eval()
    encoded.config = dict(model=encoder.MODEL, revision=encoder.REVISION, pooling='cls',normalization='l2',
                          query_prompt=encoder.QUERY_PROMPT,dimension=encoded.model.config.hidden_size,inference_dtype='float32')
    dense = Retriever(Path(asset_root)/'data/index', encoded)
    metadata = json.loads((Path(asset_root)/'evaluation/agent_pipeline_v2_1_lore_metadata.json').read_text(encoding='utf-8'))
    return V21Retriever(dense, metadata)


def record_setup(model,directory):
    from agent_pipeline_v2_2.runner import atomic_write_json
    hardware={'device':model.device,'gpu':model.torch.cuda.get_device_name(),
              'capability':list(model.torch.cuda.get_device_capability()),'cuda':model.torch.version.cuda}
    path=Path(directory)/'run_setup.json'
    if path.exists():
        old=json.loads(path.read_text(encoding='utf-8'))
        if old['hardware']!=hardware:
            raise ValueError('GPU/CUDA changed; use a new result directory')
    else:
        atomic_write_json(path,{'hardware':hardware,'model_load_seconds':model.load_seconds,
                              'model_footprint_bytes':model.model_footprint_bytes})
