import contextlib
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import entry
from judgment_protocol import validate_retrieval_answer

class EntryFeedbackTests(unittest.TestCase):
    def test_short_evidence_alias_maps_to_real_id(self):
        raw = '{"findings":[{"input_quote":"雷会疼痛。","verdict":"一致","evidence":[{"chunk_id":"E1","quote":"雷会疼痛。"}],"reason":"直接支持","new_candidates":[]}]}'
        result=validate_retrieval_answer(raw,'雷会疼痛。',[{'id':'long-id-123','text':'雷会疼痛。'}])
        self.assertEqual(result['findings'][0]['evidence'][0]['chunk_id'],'long-id-123')
        with self.assertRaises(ValueError):
            validate_retrieval_answer(raw.replace('E1','E9'),'雷会疼痛。',[{'id':'long-id-123','text':'雷会疼痛。'}])

    def test_failed_check_shows_cause_and_does_not_offer_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            lore=Path(tmp)/'lore'
            lore.mkdir()
            metadata={'lore_directory':str(lore),'sources':{},'chunks':[]}
            error={'status':'error','overall_verdict':None,'error':'引用了不存在的片段','raw_output':'bad'}
            modules={
                'index_store':SimpleNamespace(load_index=lambda p:(None,metadata),build_index=lambda *a:None),
                'encoder':SimpleNamespace(Encoder=lambda **kw:object()),
                'retrieval':SimpleNamespace(Retriever=lambda *a:SimpleNamespace(search=lambda q:[])),
                'qwen_judge':SimpleNamespace(QwenJudge=lambda d:SimpleNamespace(check=lambda *a:error)),
            }
            output=io.StringIO()
            with patch.dict(sys.modules,modules),contextlib.redirect_stdout(output):
                code=entry.main(['雷会疼痛。','--entity','雷','--category','生理结构','--preview-only','--store',str(Path(tmp)/'entries.json'),'--lore',str(lore)])
            self.assertEqual(code,2)
            self.assertIn('引用了不存在的片段',output.getvalue())
            self.assertNotIn('作者确认后可建立',output.getvalue())
            self.assertFalse((Path(tmp)/'entries.json').exists())

    def test_continuous_preview_loads_models_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            lore=Path(tmp)/'lore'
            lore.mkdir()
            metadata={'lore_directory':str(lore),'sources':{},'chunks':[]}
            loaded=[]
            def encoder(**kwargs):
                loaded.append('encoder')
                return object()
            def judge(device):
                loaded.append('judge')
                return SimpleNamespace(check=lambda *a:{'status':'ok','overall_verdict':'不确定','findings':[]})
            modules={
                'index_store':SimpleNamespace(load_index=lambda p:(None,metadata),build_index=lambda *a:None),
                'encoder':SimpleNamespace(Encoder=encoder),
                'retrieval':SimpleNamespace(Retriever=lambda *a:SimpleNamespace(search=lambda q:[])),
                'qwen_judge':SimpleNamespace(QwenJudge=judge),
            }
            with patch.dict(sys.modules,modules),patch('sys.stdin',io.StringIO('雷会疼痛。\n雷能控制时间。\n/quit\n')),contextlib.redirect_stdout(io.StringIO()):
                code=entry.main(['--interactive','--entity','雷','--category','能力','--preview-only','--lore',str(lore),'--store',str(Path(tmp)/'entries.json')])
            self.assertEqual(code,0)
            self.assertEqual(loaded,['encoder','judge'])
