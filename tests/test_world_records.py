import json
from pathlib import Path
import tempfile
import unittest
from world_records import WorldStore, validate_plan, build_catalog, presence

TEXT='第九话，雷加入晨星议会，并获得控制时间的能力。'
RECORDS=[
 {'kind':'entity','entity':'晨星议会','entity_type':'组织','category':'组织','text':'晨星议会'},
 {'kind':'timepoint','entity':'第九话','category':'时间节点','text':'第九话','time':'第九话'},
 {'kind':'relation','entity':'雷','target':'晨星议会','category':'隶属','text':'雷加入晨星议会','time':'第九话','valid_from':'第九话'},
 {'kind':'attribute','entity':'雷','category':'能力','text':'获得控制时间的能力','time':'第九话','valid_from':'第九话'},
]

class WorldRecordTests(unittest.TestCase):
    def test_general_plan_exact_input_and_time(self):
        plan=validate_plan({'records':RECORDS},TEXT)
        self.assertEqual(len(plan),4)
        self.assertEqual(plan[2]['target'],'晨星议会')
        self.assertEqual(plan[3]['valid_from'],'第九话')
        for change in [{'text':'雷创造宇宙'}, {'time':'第十话'}, {'target':'新组织'}, {'kind':'fake'}]:
            bad=dict(RECORDS[2],**change)
            with self.assertRaises(ValueError):
                validate_plan({'records':[bad]},TEXT)

    def test_rules_events_custom_types_and_unclear_subject(self):
        text='灵能来自月光。黎明城发生大地震。'
        result=validate_plan({'records':[{'kind':'rule','category':'能源规则','text':'灵能来自月光。'}, {'kind':'event','entity':'黎明城','category':'灾难','text':'黎明城发生大地震。'}, {'kind':'entity','entity':'灵能','entity_type':'自定义能量','category':'能源','text':'灵能'}]},text)
        self.assertIsNone(result[0]['entity'])
        self.assertEqual(result[2]['entity_type'],'自定义能量')
        with self.assertRaises(ValueError):
            validate_plan({'records':[{'kind':'attribute','entity':'他','category':'能力','text':'他能控制时间。'}]},'他能控制时间。')

    def test_batch_atomic_save_reload_subset_and_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=WorldStore(Path(tmp)/'world_records.json')
            plan=validate_plan({'records':RECORDS},TEXT)
            preview=store.preview(plan,TEXT,build_catalog([],[]))
            self.assertFalse(store.path.exists())
            saved=store.commit(preview,[0,2,3])
            self.assertEqual(len(saved),3)
            self.assertEqual(store.read()[0]['records'][2]['valid_from'],'第九话')
            with self.assertRaises(ValueError):
                store.commit(preview,[1])
            again=store.preview(plan,TEXT,build_catalog([],[]))
            before=store.path.read_bytes()
            with self.assertRaises(ValueError):
                store.commit(again,[0,1])
            self.assertEqual(store.path.read_bytes(),before)
            output=Path(tmp)/'generated.md'
            store.export(output)
            self.assertIn('生效起点：第九话',output.read_text(encoding='utf-8'))

    def test_full_catalog_presence_is_not_top_k_or_truth(self):
        chunks=[{'id':'c1','file':'history.md','start_line':1,'end_line':1,'text':'第九话发生大地震。','heading_path':['历史','大地震','时间']}]
        catalog=build_catalog(chunks,[])
        self.assertEqual(presence('第九话',catalog)['status'],'mentioned')
        self.assertEqual(presence('晨星议会',catalog)['status'],'not_found_in_catalog')
        self.assertNotIn('不存在',presence('晨星议会',catalog)['message'])

    def test_corruption_and_export_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'world_records.json'
            path.write_text('{}',encoding='utf-8')
            with self.assertRaises(ValueError):
                WorldStore(path).read()
            output=Path(tmp)/'manual.md'
            output.write_text('正式资料',encoding='utf-8')
            with self.assertRaises(ValueError):
                WorldStore(Path(tmp)/'empty.json').export(output)
            self.assertEqual(output.read_text(encoding='utf-8'),'正式资料')

    def test_different_records_can_share_source_quote(self):
        records=[dict(RECORDS[2],text=TEXT),dict(RECORDS[3],text=TEXT)]
        self.assertEqual(len(validate_plan({'records':records},TEXT)),2)
        with self.assertRaises(ValueError):
            validate_plan({'records':[records[0],records[0]]},TEXT)

    def test_compact_verdict_binds_original_evidence_without_model_copying(self):
        from world_records import validate_world_verdict
        row={'id':'real-id','text':'雷拥有两颗心脏。'}
        result=validate_world_verdict('{"verdict":"一致","evidence":["E1"],"reason":"证据支持"}',{'text':row['text']},[row])
        self.assertEqual(result['findings'][0]['evidence'][0],{'chunk_id':'real-id','quote':row['text']})
        for raw in ['{"verdict":"一致","evidence":[],"reason":"支持"}', '{"verdict":"一致","evidence":["E9"],"reason":"支持"}']:
            with self.assertRaises(ValueError):
                validate_world_verdict(raw,{'text':row['text']},[row])

    def test_model_payload_is_reproducible_across_processes(self):
        import os
        import subprocess
        import sys
        code="import json; from world_records import validate_plan; print(json.dumps(validate_plan({'records':[{'kind':'attribute','entity':'雷','category':'能力','text':'雷能飞。'}]},'雷能飞。'),ensure_ascii=False))"
        outputs=[subprocess.run([sys.executable,'-X','utf8','-c',code],env=dict(os.environ,PYTHONHASHSEED=str(seed)),capture_output=True,text=True,encoding='utf-8',check=True).stdout for seed in (1,2)]
        self.assertEqual(outputs[0],outputs[1])
