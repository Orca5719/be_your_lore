"""Qwen learning experiment: chat template -> tokens -> local generation."""
import argparse
import json
from pathlib import Path
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL = 'Qwen/Qwen3-4B-Instruct-2507'
REVISION = 'cdbee75f17c01a7cc42f958dc650907174af0554'
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--prompt', default='请用一句中文解释：什么是世界观设定？')
    args = parser.parse_args()
    if not args.prompt.strip():
        parser.error('输入不能为空')
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA 不可用，请选择 CPU')
    shared = dict(revision=REVISION, cache_dir=ROOT / '.cache' / 'huggingface', local_files_only=not args.download)
    print('加载 tokenizer...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, **shared)
    messages = [{'role': 'system', 'content': '你是中文助手，请简洁回答。'}, {'role': 'user', 'content': args.prompt}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors='pt', return_dict=True)
    if inputs['input_ids'].shape[1] + 128 > 4096:
        parser.error('输入与输出预算合计超过 4096 tokens，不会截断输入')
    kwargs = dict(dtype=torch.bfloat16, device_map={'': device}, use_safetensors=True)
    if device == 'cuda':
        kwargs['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    else:
        torch.set_num_threads(4)
    print(f'加载模型：{device}，GPU NF4 / CPU BF16；首次下载可能较久...', flush=True)
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(MODEL, **shared, **kwargs).eval()
    load_seconds = time.perf_counter() - started
    inputs = inputs.to(device)
    if device == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, do_sample=False, max_new_tokens=128, use_cache=True)
    if device == 'cuda':
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    generated = output[0, inputs['input_ids'].shape[1]:].tolist()
    answer = tokenizer.decode(generated, skip_special_tokens=True)
    if not answer.strip():
        raise RuntimeError('模型生成了空回答')
    report = dict(model=MODEL, revision=REVISION, device=device,
        quantization='NF4 double quant' if device == 'cuda' else 'none', compute_dtype='bfloat16',
        quantized_linear_count=sum(type(m).__name__ == 'Linear4bit' for m in model.modules()),
        chat_template_text=rendered, input_shape=list(inputs['input_ids'].shape),
        input_ids=inputs['input_ids'][0].tolist(), tokens=tokenizer.convert_ids_to_tokens(inputs['input_ids'][0].tolist()),
        attention_mask=inputs['attention_mask'][0].tolist(), output_shape=list(output.shape),
        generated_ids=generated, answer=answer, output_limit_reached=len(generated) == 128,
        load_seconds=load_seconds, generation_seconds=seconds,
        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if device == 'cuda' else None,
        note='单次冷生成实验，不是预热后的性能 benchmark，也不是一致性准确率评测。')
    (ROOT / 'reports').mkdir(exist_ok=True)
    (ROOT / 'reports' / 'experiment05_qwen.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
