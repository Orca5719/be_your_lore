"""按 Markdown 标题和段落分块，使用真实 tokenizer 检查完整编码输入。"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import re

@dataclass
class Chunk:
    id: str
    text: str
    file: str
    start_line: int
    end_line: int
    heading_path: list[str]
    start_offset: int
    end_offset: int
    embedding_text: str
    token_count: int


def load_chunks(directory, tokenizer, max_tokens=512):
    root = Path(directory).resolve()
    if not root.is_dir():
        raise ValueError(f'资料目录不存在：{root}')
    if not 3 <= max_tokens <= 512:
        raise ValueError('max_tokens 必须在 3 到 512 之间')
    chunks = []
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in {'.md', '.txt'}:
            continue
        # 用逻辑文本偏移：保留换行，不做 strip 或模型解码重建正文。
        source = path.read_text(encoding='utf-8-sig')
        relative = path.relative_to(root).as_posix()
        headings = []
        paragraph_start = None
        paragraph_end = None
        offset = 0
        in_fence = False

        def emit(start, end):
            if start is None:
                return
            titles = [title for _, title in headings]
            prefix = ' > '.join(titles) + '\n' if titles else ''

            def count(a, b):
                return len(tokenizer.encode(prefix + source[a:b], add_special_tokens=True, truncation=False))

            if len(tokenizer.encode(prefix, add_special_tokens=True)) >= max_tokens:
                raise ValueError(f'{relative}: 标题占满 token 预算，请缩短标题')
            position = start
            while position < end:
                if count(position,end) <= max_tokens:
                    stop = end
                else:
                    # 每个候选边界都检查，避免假设 token 数随字符数严格单调。
                    stops = [position+m.end() for m in re.finditer(r'[。！？!?；;](?:[”’"\u300d\u300f]*)|\n',source[position:end])]
                    fitting = [s for s in stops if count(position,s) <= max_tokens]
                    if fitting:
                        stop = max(fitting)
                    else:
                        stop = position
                        for candidate in range(position+1,end+1):
                            if count(position,candidate) > max_tokens:
                                break
                            stop = candidate
                        if stop == position:
                            raise ValueError(f'{relative}: 一个字符也无法放入 token 预算')
                text = source[position:stop]
                identity = f'{relative}\0{position}\0{stop}\0{text}\0{titles}'
                chunks.append(Chunk(
                    hashlib.sha256(identity.encode()).hexdigest()[:20], text, relative,
                    source.count('\n',0,position)+1,source.count('\n',0,stop-1)+1,
                    titles.copy(),position,stop,prefix+text,count(position,stop)))
                if stop == end:
                    break
                next_position = stop
                # 只有完整句子且前后都能推进时，重叠上一块的最后一句。
                boundaries = [position+m.end() for m in re.finditer(r'[。！？!?；;](?:[”’"\u300d\u300f]*)|\n',source[position:stop])]
                if len(boundaries) >= 2 and boundaries[-1] == stop:
                    overlap = boundaries[-2]
                    next_boundary = re.search(r'[。！？!?；;]|\n',source[stop:end])
                    next_end = stop+next_boundary.end() if next_boundary else end
                    if overlap > position and count(overlap,next_end) <= max_tokens:
                        next_position = overlap
                position = next_position

        for line in source.splitlines(keepends=True):
            content = line.rstrip('\r\n')
            fence = re.match(r'^\s{0,3}(`{3,}|~{3,})',content) if path.suffix.lower()=='.md' else None
            heading = re.match(r'^\s{0,3}(#{1,6})\s+(.+?)\s*$',content) if path.suffix.lower()=='.md' and not in_fence else None
            if heading or not content.strip():
                emit(paragraph_start,paragraph_end)
                paragraph_start = paragraph_end = None
                if heading:
                    level = len(heading[1])
                    headings = [(n,t) for n,t in headings if n<level]
                    headings.append((level,re.sub(r'\s+#+\s*$','',heading[2])))
            else:
                if paragraph_start is None:
                    paragraph_start = offset
                paragraph_end = offset+len(content)
            if fence:
                in_fence = not in_fence
            offset += len(line)
        emit(paragraph_start,paragraph_end)
    if not chunks:
        raise ValueError('资料目录中没有可编码的 Markdown/TXT 正文')
    return chunks
