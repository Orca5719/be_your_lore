import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from transformers import AutoTokenizer
from documents import load_chunks

ROOT = Path(__file__).resolve().parents[1]

class ChunkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-small-zh-v1.5', cache_dir=ROOT/'.cache/huggingface', local_files_only=True)

    def test_headings_and_source_offsets(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            text = '# 人物\n## 雷\n\n雷有两颗心脏。\n左侧寄宿亚巴顿。\n\n### 经历\n曾经历地震。\n'
            (root/'人物.md').write_text(text, encoding='utf-8')
            chunks = load_chunks(root, self.tokenizer)
            self.assertEqual(len(chunks), 2)
            self.assertEqual(chunks[0].heading_path, ['人物','雷'])
            self.assertEqual((chunks[0].start_line,chunks[0].end_line), (4,5))
            for chunk in chunks:
                self.assertEqual(text[chunk.start_offset:chunk.end_offset],chunk.text)
            self.assertEqual([c.id for c in chunks], [c.id for c in load_chunks(root,self.tokenizer)])

    def test_long_paragraph_coverage_and_overlap(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            text = '# 雷\n\n' + ''.join(f'第{i}次事件雷左侧心脏出现疼痛。' for i in range(20))
            (root/'a.md').write_text(text, encoding='utf-8')
            chunks = load_chunks(root,self.tokenizer,max_tokens=64)
            self.assertGreater(len(chunks),1)
            for chunk in chunks:
                self.assertLessEqual(chunk.token_count,64)
                self.assertEqual(text[chunk.start_offset:chunk.end_offset],chunk.text)
            self.assertEqual(chunks[0].start_offset, len('# 雷\n\n'))
            self.assertEqual(chunks[-1].end_offset,len(text))
            for a,b in zip(chunks,chunks[1:]):
                self.assertLess(b.start_offset,a.end_offset)

    def test_single_long_sentence_and_recursive_txt(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'sub').mkdir()
            text = '雷的左心脏' * 100
            (root/'sub/a.TXT').write_text(text,encoding='utf-8')
            chunks = load_chunks(root,self.tokenizer,max_tokens=32)
            self.assertEqual(''.join(c.text for c in chunks),text)
            self.assertTrue(all(c.token_count <= 32 for c in chunks))
            self.assertEqual(chunks[0].file,'sub/a.TXT')

    def test_empty_and_title_budget_errors(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                load_chunks(root,self.tokenizer)
            (root/'a.md').write_text('# '+ '很长标题'*30+'\n正文',encoding='utf-8')
            with self.assertRaises(ValueError):
                load_chunks(root,self.tokenizer,max_tokens=32)

if __name__ == '__main__':
    unittest.main()
