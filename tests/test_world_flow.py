import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import world_entry
from world_records import WorldStore
from test_world_records import TEXT,RECORDS

class WorldFlowTests(unittest.TestCase):
    def run_case(self,response,fail_sync=False,partial=False):
        with tempfile.TemporaryDirectory() as tmp:
            lore=Path(tmp)/'lore'
            lore.mkdir()
            store=Path(tmp)/'world.json'
            builds=[]
            def build(*args):
                builds.append(True)
                if fail_sync:
                    raise OSError('同步测试失败')
                return Path('version')
            judge=SimpleNamespace(extract_world=lambda *a:{'records':RECORDS,'rejected':[{'record_index':5,'error':'非法目标'}] if partial else []},check_world=lambda text,records,results:{'status':'ok','findings':[{'input_quote':text,'verdict':'不确定','reason':'未提供依据','evidence':[]}],'evidence':[]})
            modules={'index_store':SimpleNamespace(load_index=lambda p:(None,{'lore_directory':str(lore),'sources':{},'chunks':[]}),build_index=build),
                     'encoder':SimpleNamespace(Encoder=lambda **kwargs:object()),
                     'retrieval':SimpleNamespace(Retriever=lambda *a:SimpleNamespace(search=lambda q:[])),
                     'qwen_judge':SimpleNamespace(QwenJudge=lambda *a:judge)}
            output=io.StringIO()
            with patch.dict(sys.modules,modules),patch('sys.stdin',io.StringIO(response+'\n')),contextlib.redirect_stdout(output):
                code=world_entry.main([TEXT,'--store',str(store),'--legacy-store',str(Path(tmp)/'legacy.json'),'--lore',str(lore),'--json'])
            rows=[json.loads(line) for line in output.getvalue().splitlines()]
            saved=WorldStore(store).read()[0]['records'] if store.exists() else []
            return code,rows,saved,builds

    def test_cancel_has_no_persistence_or_rebuild(self):
        code,rows,saved,builds=self.run_case('取消')
        self.assertEqual(code,0)
        self.assertEqual(rows[-1]['status'],'cancelled')
        self.assertEqual(saved,[])
        self.assertEqual(builds,[])

    def test_selected_batch_is_confirmed_once(self):
        code,rows,saved,builds=self.run_case('确认保存 1,3')
        self.assertEqual(code,0)
        self.assertEqual(rows[-1]['count'],2)
        self.assertEqual({item['kind'] for item in saved},{'entity','relation'})
        self.assertEqual(len(builds),1)

    def test_sync_failure_does_not_claim_unsaved_or_repeat_commit(self):
        code,rows,saved,builds=self.run_case('确认保存 1,3',True)
        self.assertEqual(code,2)
        self.assertEqual(rows[-1]['status'],'saved')
        self.assertEqual(rows[-1]['index_status'],'error')
        self.assertEqual(len(saved),2)
        self.assertEqual(len(builds),1)

    def test_partial_preview_requires_explicit_selection(self):
        code,rows,saved,builds=self.run_case('确认保存',partial=True)
        self.assertEqual(code,2)
        self.assertEqual(rows[0]['status'],'partial_extraction')
        self.assertEqual(saved,[])
        code,rows,saved,builds=self.run_case('确认保存 1,3',partial=True)
        self.assertEqual(code,0)
        self.assertEqual(len(saved),2)

    def test_judge_receives_original_context_without_persisting_it(self):
        from entries import EntryStore
        with tempfile.TemporaryDirectory() as tmp:
            captured=[]
            def check(text,records,results):
                captured.extend(records)
                return {'status':'ok','findings':[{'input_quote':text,'verdict':'不确定','reason':'测试','evidence':[]}]}
            judge=SimpleNamespace(extract_world=lambda *a:{'records':RECORDS},check_world=check)
            store=WorldStore(Path(tmp)/'world.json')
            preview=world_entry.prepare(store,EntryStore(Path(tmp)/'old.json'),[],TEXT,judge,SimpleNamespace(search=lambda q:[]))
            self.assertTrue(all(row['source_input']==TEXT for row in captured))
            self.assertTrue(all('source_input' not in item['proposed'] for item in preview['items']))
