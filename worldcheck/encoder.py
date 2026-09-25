"""公开可观察的 BGE 编码流程：tokenizer -> Transformer -> CLS -> L2。"""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

MODEL = 'BAAI/bge-small-zh-v1.5'
REVISION = '7999e1d3359715c523056ef9478215996d62a620'
QUERY_PROMPT = '为这个句子生成表示以用于检索相关文章：'

class Encoder:
    def __init__(self, device='auto', offline=True, precision='float32'):
        if device not in {'auto','cpu','cuda'}:
            raise ValueError('设备必须为 auto、cpu 或 cuda')
        self.device = 'cuda' if device=='auto' and torch.cuda.is_available() else device
        if self.device=='auto':
            self.device='cpu'
        if self.device=='cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA 不可用，请选择 CPU')
        if precision not in {'float32','float16'} or (precision=='float16' and self.device!='cuda'):
            raise ValueError('precision 必须为 float32，或 CUDA 上的 float16')
        dtype=torch.float16 if precision=='float16' else torch.float32
        cache = Path(__file__).resolve().parent/'.cache/huggingface'
        options = dict(cache_dir=cache,revision=REVISION,local_files_only=offline)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL,**options)
        self.model = AutoModel.from_pretrained(MODEL,use_safetensors=True,**options).to(device=self.device,dtype=dtype).eval()
        self.config = {'model':MODEL,'revision':REVISION,'pooling':'cls','normalization':'l2','query_prompt':QUERY_PROMPT,'dimension':self.model.config.hidden_size,'inference_dtype':precision}

    def _checked_texts(self,texts):
        texts=list(texts)
        if not texts or any(not isinstance(t,str) or not t.strip() for t in texts):
            raise ValueError('编码输入不能为空')
        if any(len(self.tokenizer.encode(t,add_special_tokens=True,verbose=False))>512 for t in texts):
            raise ValueError('完整编码输入超过 512 tokens，请先分块')
        return texts

    def _forward(self,texts):
        inputs=self.tokenizer(texts,padding=True,truncation=False,return_tensors='pt')
        inputs={k:v.to(self.device) for k,v in inputs.items()}
        with torch.inference_mode():
            hidden=self.model(**inputs).last_hidden_state
            cls=hidden[:,0,:]
            vectors=F.normalize(cls.float(),p=2,dim=1)
        return inputs,hidden,cls,vectors

    def encode_passages(self,texts,batch_size=16):
        texts=self._checked_texts(texts)
        if isinstance(batch_size,bool) or not isinstance(batch_size,int) or batch_size<1:
            raise ValueError('batch_size 必须为正整数')
        batches=[]
        for start in range(0,len(texts),batch_size):
            _,_,_,vectors=self._forward(texts[start:start+batch_size])
            batches.append(vectors.cpu().numpy())
        return np.concatenate(batches).astype(np.float32,copy=False)

    def inspect_text(self,text,mode='passage'):
        if mode not in {'passage','query'}:
            raise ValueError('mode 必须为 passage 或 query')
        self._checked_texts([text])
        encoding_text=QUERY_PROMPT+text if mode=='query' else text
        inputs,hidden,cls,vectors=self._forward(self._checked_texts([encoding_text]))
        ids=inputs['input_ids'][0].cpu().tolist()
        return {
            'type':'text','mode':mode,'input_text':text,'encoding_text':encoding_text,
            'config':self.config.copy(),'device':self.device,'dtype':str(vectors.dtype),
            'tokens':self.tokenizer.convert_ids_to_tokens(ids),'input_ids':ids,
            'attention_mask':inputs['attention_mask'][0].cpu().tolist(),
            'input_shape':list(inputs['input_ids'].shape),'hidden_shape':list(hidden.shape),
            'cls_shape':list(cls.shape),'cls_head':cls[0,:8].cpu().tolist(),
            'normalized_head':vectors[0,:8].cpu().tolist(),
            'raw_norm':float(cls[0].norm()),'normalized_norm':float(vectors[0].norm()),
        }

    def encode_queries(self,texts,batch_size=16):
        texts=list(texts)
        if not texts or any(not isinstance(t,str) or not t.strip() for t in texts):
            raise ValueError('查询不能为空')
        return self.encode_passages([QUERY_PROMPT+t for t in texts],batch_size)
