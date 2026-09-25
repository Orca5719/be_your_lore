"""精确检索：单位向量矩阵点积，然后稳定降序排序。"""
import numpy as np
from index_store import load_index, validate

NOTICE='相似度表示相对相关性，不是事实正确率或矛盾置信度；无关查询也可能返回片段。'


def rank_chunks(vectors,chunks,query_vector,k=5):
    if isinstance(k,bool) or not isinstance(k,int) or k<1:
        raise ValueError('k 必须为正整数')
    validate(vectors,{'chunks':chunks,'config':{}})
    q=np.asarray(query_vector)
    if q.shape!=(vectors.shape[1],) or not np.isfinite(q).all() or not np.isclose(np.linalg.norm(q),1,atol=1e-5):
        raise ValueError('查询向量维度不匹配，或不是有效单位向量')
    scores=vectors @ q
    order=np.argsort(-scores,kind='stable')[:min(k,len(chunks))]
    return [dict(chunks[int(row)],score=float(scores[row])) for row in order]


class Retriever:
    def __init__(self,index_directory,encoder,max_tokens=512):
        self.encoder=encoder
        expected=dict(encoder.config,chunker={'version':1,'max_tokens':max_tokens,'overlap':'last_sentence_when_fits'})
        self.vectors,self.metadata=load_index(index_directory,expected_config=expected)

    def search(self,query,k=5):
        if not isinstance(query,str) or not query.strip():
            raise ValueError('查询不能为空')
        if isinstance(k,bool) or not isinstance(k,int) or k<1:
            raise ValueError('k 必须为正整数')
        vector=self.encoder.encode_queries([query])[0]
        return rank_chunks(self.vectors,self.metadata['chunks'],vector,k)


def search(query,index_directory,encoder,k=5,max_tokens=512):
    """返回包含原文、来源、片段 ID 和 cosine score 的结果列表。"""
    return Retriever(index_directory,encoder,max_tokens).search(query,k)
