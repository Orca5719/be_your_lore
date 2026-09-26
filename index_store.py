"""每个版本保存一对向量/元数据，通过原子替换 CURRENT 发布。"""
import hashlib
import json
import os
from pathlib import Path
import re
import uuid
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(vectors,metadata):
    chunks=metadata.get('chunks')
    if not isinstance(chunks,list) or not chunks or not isinstance(metadata.get('config'),dict):
        raise ValueError('索引元数据无效，请重建')
    if vectors.ndim!=2 or vectors.shape[0]!=len(chunks) or vectors.shape[1]<1 or vectors.dtype!=np.float32:
        raise ValueError('向量形状、类型或片段数量不匹配，请重建')
    ids=[c.get('id') for c in chunks if isinstance(c,dict)]
    if len(ids)!=len(chunks) or any(not isinstance(i,str) or not i for i in ids) or len(set(ids))!=len(ids):
        raise ValueError('片段 ID 无效或重复，请重建')
    dimension=metadata['config'].get('dimension')
    if dimension is not None and dimension!=vectors.shape[1]:
        raise ValueError('模型维度与向量不匹配，请重建')
    if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-5):
        raise ValueError('向量不是有效单位向量，请重建')


def save_index(directory,vectors,metadata):
    validate(vectors,metadata)
    root=Path(directory)
    version=uuid.uuid4().hex
    target=root/'versions'/version
    target.mkdir(parents=True)
    np.save(target/'embeddings.npy',vectors,allow_pickle=False)
    info=dict(metadata,schema_version=1,shape=list(vectors.shape),storage_dtype='float32')
    (target/'metadata.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    manifest={name:digest(target/name) for name in ['embeddings.npy','metadata.json']}
    (target/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    # 发布前读取磁盘文件验证；失败不改变当前版本。
    read_version(target)
    pointer=root/f'.CURRENT-{version}'
    pointer.write_text(version,encoding='ascii')
    os.replace(pointer,root/'CURRENT')
    return target


def read_version(target):
    try:
        manifest=json.loads((target/'manifest.json').read_text(encoding='utf-8'))
        for name in ['embeddings.npy','metadata.json']:
            if digest(target/name)!=manifest[name]:
                raise ValueError('索引文件摘要不匹配，请重建')
        vectors=np.load(target/'embeddings.npy',allow_pickle=False)
        metadata=json.loads((target/'metadata.json').read_text(encoding='utf-8'))
        if metadata.get('schema_version')!=1 or metadata.get('shape')!=list(vectors.shape):
            raise ValueError('索引版本或形状无效，请重建')
        validate(vectors,metadata)
        return vectors,metadata
    except (OSError,ValueError,KeyError,TypeError,AttributeError) as error:
        raise ValueError(f'无法加载索引，请重建：{error}') from error


def load_index(directory,expected_config=None):
    root=Path(directory)
    try:
        version=(root/'CURRENT').read_text(encoding='ascii').strip()
    except (OSError,UnicodeError) as error:
        raise ValueError('缺少有效索引，请先建库') from error
    if not re.fullmatch(r'[0-9a-f]{32}',version):
        raise ValueError('索引版本指针无效，请重建')
    vectors,metadata=read_version(root/'versions'/version)
    if expected_config is not None and metadata['config']!=expected_config:
        raise ValueError('模型或分块配置不匹配，请重建')
    return vectors,metadata


def build_index(lore_directory,index_directory,encoder,*,max_tokens=512,batch_size=16):
    from dataclasses import asdict
    from documents import load_chunks
    root=Path(lore_directory).resolve()
    def sources():
        return {p.relative_to(root).as_posix():digest(p) for p in sorted(root.rglob('*')) if p.is_file() and p.suffix.lower() in {'.md','.txt'}}
    before=sources()
    chunks=load_chunks(root,encoder.tokenizer,max_tokens)
    vectors=encoder.encode_passages([c.embedding_text for c in chunks],batch_size)
    if sources()!=before:
        raise ValueError('建库期间资料发生变化，请重新运行')
    metadata={'config':dict(encoder.config,chunker={'version':1,'max_tokens':max_tokens,'overlap':'last_sentence_when_fits'}),'sources':before,'lore_directory':str(root),'batch_size':batch_size,'chunks':[asdict(c) for c in chunks]}
    return save_index(index_directory,vectors,metadata)
