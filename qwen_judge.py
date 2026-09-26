"""Offline Qwen evidence judge. Retrieval and persistence stay outside this class."""
from pathlib import Path
from judgment_protocol import validate_retrieval_answer

ROOT = Path(__file__).resolve().parent
MODEL = 'Qwen/Qwen3-4B-Instruct-2507'
REVISION = 'cdbee75f17c01a7cc42f958dc650907174af0554'

class GenerationError(ValueError):
    def __init__(self, message, raw_output):
        super().__init__(message)
        self.raw_output = raw_output


class QwenJudge:
    def __init__(self, device='auto'):
        import time
        started = time.perf_counter()
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        self.torch = torch
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        if self.device not in ('cpu','cuda') or (self.device == 'cuda' and not torch.cuda.is_available()):
            raise ValueError('设备不可用，请选择 CPU 或可用 CUDA')
        # Resolve pinned local snapshot first; using a filesystem path avoids auxiliary Hub requests.
        snapshot = snapshot_download(MODEL, revision=REVISION, cache_dir=ROOT/'.cache/huggingface', local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        kwargs = dict(dtype=torch.bfloat16, device_map={'': self.device}, local_files_only=True, use_safetensors=True)
        if self.device == 'cuda':
            kwargs['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
        else:
            torch.set_num_threads(4)
        self.model = AutoModelForCausalLM.from_pretrained(snapshot, **kwargs).eval()
        self.prompt = (ROOT/'prompts/consistency_v4.txt').read_text(encoding='utf-8-sig')
        self.model_footprint_bytes = self.model.get_memory_footprint()
        self.model_cuda_allocated_bytes = torch.cuda.memory_allocated() if self.device == 'cuda' else None
        self.load_seconds = time.perf_counter()-started
        self.last_generation = {}

    def _generate(self, messages, max_new_tokens=768):
        self.last_generation = {}
        inputs = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt').to(self.device)
        count = inputs['input_ids'].shape[1]
        if count + max_new_tokens > 4096:
            raise ValueError('输入和证据超过 4096 tokens 预算，不会截断')
        import time
        if self.device == 'cuda':
            self.torch.cuda.synchronize()
        from model_metrics import TokenTiming
        token_timing = TokenTiming()
        started = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, do_sample=False, temperature=1.0, top_p=1.0, top_k=50, max_new_tokens=max_new_tokens, use_cache=True, streamer=token_timing)
        if self.device == 'cuda':
            self.torch.cuda.synchronize()
        self.last_generation = dict(seconds=time.perf_counter()-started,input_tokens=count,generated_tokens=output.shape[1]-count, **token_timing.metrics(started))
        ids = output[0,count:].tolist()
        raw = self.tokenizer.decode(ids, skip_special_tokens=True)
        eos = self.model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos,int) else eos or []
        if len(ids) == max_new_tokens and ids[-1] not in eos:
            raise GenerationError('生成达到上限且未结束',raw)
        return raw

    def classify(self, text, known_entities, categories, overrides=None):
        import json
        from classification_protocol import validate_classification
        if not isinstance(text,str) or not text.strip():
            raise ValueError('输入不能为空')
        prompt = (ROOT/'prompts/classification_v1.txt').read_text(encoding='utf-8-sig')
        messages = [{'role':'system','content':prompt}, {'role':'user','content':json.dumps(dict(new_text=text, known_entities=known_entities, allowed_categories=categories, overrides=overrides or {}),ensure_ascii=False)}]
        raw = self._generate(messages, max_new_tokens=192)
        return dict(validate_classification(raw,text,categories,overrides),raw_output=raw,prompt_version='classification-v1',timing=dict(self.last_generation))

    def check(self, text, results):
        import json
        if not isinstance(text, str) or not text.strip():
            raise ValueError('输入不能为空')
        evidence = {'E'+str(i): x['text'] for i,x in enumerate(results,1)}
        messages = [{'role':'system','content':self.prompt}, {'role':'user','content':json.dumps({'new_text':text,'evidence':evidence},ensure_ascii=False)}]
        raw = ''
        try:
            raw = self._generate(messages,max_new_tokens=512)
            answer = validate_retrieval_answer(raw,text,results)
        except ValueError as exc:
            raw = getattr(exc,'raw_output',raw)
            answer = dict(status='error',overall_verdict=None,error=str(exc))
        return dict(answer,raw_output=raw, evidence=results, model=MODEL, revision=REVISION, prompt_version='consistency-v4', device=self.device,timing=dict(getattr(self,'last_generation',{})))



    def extract_world(self,text,catalog):
        import json
        from world_extraction import source_spans, decode_records
        if not isinstance(text,str) or not text.strip():
            raise ValueError('输入不能为空')
        spans=source_spans(text)
        prompt=(ROOT/'prompts/world_extraction_v4.txt').read_text(encoding='utf-8-sig')
        messages=[{'role':'system','content':prompt},{'role':'user','content':json.dumps(dict(new_text=text,source_spans=spans,existing_names=catalog['names'],existing_times=catalog['times']),ensure_ascii=False)}]
        raw=self._generate(messages,max_new_tokens=1024)
        try:
            value=json.loads(raw)
        except ValueError as exc:
            raise GenerationError('多条归档输出不是有效 JSON，未保存',raw) from exc
        try:
            decoded=decode_records(value,text,spans,recover=True)
        except ValueError as exc:
            raise GenerationError(str(exc),raw) from exc
        return dict(decoded,raw_output=raw,prompt_version='world-extraction-v4',timing=dict(self.last_generation))

    def check_world(self,text,records,results):
        import json
        from world_records import validate_world_verdict
        if len(records)!=1:
            raise ValueError('通用判断接口一次检查一个独立条目')
        evidence={'E'+str(i):row['text'] for i,row in enumerate(results,1)}
        instructions=(ROOT/'prompts/world_review_v2.txt').read_text(encoding='utf-8-sig')
        messages=[{'role':'system','content':instructions},{'role':'user','content':json.dumps(dict(new_text=text,claim_context=records[0],evidence=evidence),ensure_ascii=False)}]
        raw=''
        try:
            raw=self._generate(messages,max_new_tokens=256)
            result=validate_world_verdict(raw,records[0],results)
        except ValueError as exc:
            raw=getattr(exc,'raw_output',raw)
            result=dict(status='error',overall_verdict=None,error=str(exc))
        return dict(result,raw_output=raw,evidence=results,prompt_version='world-review-v2',timing=dict(self.last_generation))

