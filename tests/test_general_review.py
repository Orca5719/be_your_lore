import io
import json
from pathlib import Path
import tempfile
import unittest
from world_records import WorldStore
from entries import EntryStore
from world_entry import prepare
from test_world_records import TEXT,RECORDS

class GeneralReviewTests(unittest.TestCase):
    def test_each_claim_requires_its_own_verdict(self):
        with tempfile.TemporaryDirectory() as tmp:
            class Judge:
                def extract_world(self,text,catalog):
                    return {'records':RECORDS}
                def check_world(self,text,records,results):
                    return {'status':'ok','findings':[{'input_quote':TEXT,'verdict':'矛盾','reason':'混合总判断','evidence':[]}],'evidence':[]}
            class Retriever:
                def search(self,query):
                    return []
            preview=prepare(WorldStore(Path(tmp)/'world.json'),EntryStore(Path(tmp)/'legacy.json'),[],TEXT,Judge(),Retriever())
            self.assertEqual(preview['status'],'check_failed')
            facts=[item for item in preview['items'] if item['proposed']['kind'] in ('attribute','relation')]
            self.assertTrue(all(item['review']['status']=='error' for item in facts))
            self.assertFalse((Path(tmp)/'world.json').exists())

    def test_distinct_verdicts_do_not_leak_between_claims(self):
        with tempfile.TemporaryDirectory() as tmp:
            class Judge:
                def extract_world(self,text,catalog):
                    return {'records':RECORDS}
                def check_world(self,text,records,results):
                    return {'status':'ok','findings':[{'input_quote':record['text'],'verdict':'不确定' if record['category']=='能力' else '矛盾','reason':'待核对','evidence':[]} for record,verdict in zip(records,['矛盾','不确定'])],'evidence':[]}
            class Retriever:
                def search(self,query):
                    return []
            preview=prepare(WorldStore(Path(tmp)/'world.json'),EntryStore(Path(tmp)/'legacy.json'),[],TEXT,Judge(),Retriever())
            self.assertEqual(preview['status'],'pending_confirmation')
            self.assertEqual(preview['items'][2]['review']['verdict'],'矛盾')
            self.assertEqual(preview['items'][3]['review']['verdict'],'不确定')

