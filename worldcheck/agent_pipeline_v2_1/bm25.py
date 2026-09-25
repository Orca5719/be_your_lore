"""Dependency-free Chinese lexical retrieval using standard BM25."""

from __future__ import annotations

from collections import Counter
import math
import re
import unicodedata


TOKEN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+|[a-z0-9_]+", re.IGNORECASE)


def tokenize_zh(text: str) -> list[str]:
    if not isinstance(text, str):
        raise ValueError("分词输入必须是文字")
    normalized = unicodedata.normalize("NFKC", text).lower()
    tokens = []
    for match in TOKEN_RE.finditer(normalized):
        value = match.group(0)
        if re.fullmatch(r"[\u3400-\u4dbf\u4e00-\u9fff]+", value):
            tokens.extend(value)
            tokens.extend(value[index : index + 2] for index in range(len(value) - 1))
        else:
            tokens.append(value)
    return tokens


def _document_text(chunk: dict) -> str:
    headings = chunk.get("heading_path") or []
    return "\n".join([*headings, chunk.get("text", "")])


class BM25Retriever:
    def __init__(self, chunks: list[dict], k1: float = 1.5, b: float = 0.75):
        if not isinstance(chunks, list) or not chunks:
            raise ValueError("BM25 chunks不能为空")
        if not isinstance(k1, (int, float)) or isinstance(k1, bool) or k1 <= 0:
            raise ValueError("BM25 k1必须为正数")
        if not isinstance(b, (int, float)) or isinstance(b, bool) or not 0 <= b <= 1:
            raise ValueError("BM25 b必须在0到1之间")
        ids = [chunk.get("id") for chunk in chunks]
        if any(not isinstance(value, str) or not value for value in ids) or len(ids) != len(set(ids)):
            raise ValueError("BM25 chunk id无效或重复")
        self.chunks = chunks
        self.k1 = float(k1)
        self.b = float(b)
        self.tokens = [tokenize_zh(_document_text(chunk)) for chunk in chunks]
        self.term_frequencies = [Counter(row) for row in self.tokens]
        self.lengths = [len(row) for row in self.tokens]
        self.average_length = sum(self.lengths) / len(self.lengths)
        document_frequency = Counter()
        for row in self.term_frequencies:
            document_frequency.update(row.keys())
        count = len(chunks)
        self.idf = {
            term: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }
        self.index_by_id = {chunk["id"]: index for index, chunk in enumerate(chunks)}

    def _score(self, query_terms: list[str], index: int) -> float:
        frequencies = self.term_frequencies[index]
        length = self.lengths[index]
        normalization = self.k1 * (1.0 - self.b + self.b * length / self.average_length) if self.average_length else self.k1
        score = 0.0
        for term in set(query_terms):
            frequency = frequencies.get(term, 0)
            if frequency:
                score += self.idf.get(term, 0.0) * (frequency * (self.k1 + 1.0)) / (frequency + normalization)
        return score

    def search_with_report(self, query: str, k: int = 5, candidate_ids: list[str] | None = None) -> dict:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("查询不能为空")
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k必须为正整数")
        if candidate_ids is None:
            indices = list(range(len(self.chunks)))
        else:
            if not isinstance(candidate_ids, list) or not candidate_ids or len(candidate_ids) != len(set(candidate_ids)):
                raise ValueError("candidate_ids必须是非空且不重复的数组")
            unknown = [value for value in candidate_ids if value not in self.index_by_id]
            if unknown:
                raise ValueError("candidate id不属于BM25索引：" + "、".join(unknown))
            allowed = set(candidate_ids)
            indices = [index for index, chunk in enumerate(self.chunks) if chunk["id"] in allowed]
        query_terms = tokenize_zh(query)
        scored = [(index, self._score(query_terms, index)) for index in indices]
        scored.sort(key=lambda value: (-value[1], value[0]))
        results = []
        for rank, (index, score) in enumerate(scored[: min(k, len(scored))], 1):
            frequencies = self.term_frequencies[index]
            matched = list(dict.fromkeys(term for term in query_terms if term in frequencies))
            results.append({
                **self.chunks[index],
                "score": float(score),
                "lexical_score": float(score),
                "rank": rank,
                "matched_terms": matched,
            })
        return {
            "schema_version": "agent-pipeline-v2.1-bm25-result-v1",
            "method": "bm25",
            "query_token_count": len(query_terms),
            "candidate_filter_applied": candidate_ids is not None,
            "candidate_count": len(indices),
            "corpus_count": len(self.chunks),
            "config": {"tokenizer": "zh-unigram-bigram-v1", "k1": self.k1, "b": self.b},
            "results": results,
        }

    def search(self, query: str, k: int = 5, candidate_ids: list[str] | None = None) -> list[dict]:
        return self.search_with_report(query, k=k, candidate_ids=candidate_ids)["results"]

