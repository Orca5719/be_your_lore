import unittest
from world_extraction import source_spans, decode_records

class ExtractionTests(unittest.TestCase):
    def test_omitted_subject_stays_verbatim(self):
        text='第18话，茱莉亚入职星球日报，并结识克拉克。'
        spans=source_spans(text)
        self.assertEqual(spans['S3'],'并结识克拉克')
        records=decode_records({'records':[{'k':'relation','e':'茱莉亚','c':'结识','x':'克拉克','s':'S3','t':'第18话'}]},text,spans)
        self.assertEqual(records[0]['text'],'并结识克拉克')
        self.assertEqual(records[0]['entity'],'茱莉亚')

    def test_invalid_references_and_invented_names_rejected(self):
        text='雷加入议会。'
        for row in [{'k':'event','c':'经历','s':'S99'}, {'k':'relation','c':'隶属','e':'雷','x':'银河议会','s':'S1'}, {'k':'event','c':'经历','s':'S1','text':'雷统治世界'}]:
            with self.assertRaises(ValueError):
                decode_records({'records':[row]},text,source_spans(text))

    def test_structure_quotes_derived_and_long_sentence_kept(self):
        text='第18话，茱莉亚入职星球日报，并结识克拉克。'
        result=decode_records({'records':[{'k':'entity','e':'茱莉亚','y':'人物'}, {'k':'timepoint','e':'第18话'}]},text,source_spans(text))
        self.assertEqual(result[0]['text'],'茱莉亚')
        self.assertEqual(result[1]['time'],'第18话')
        clause='甲'*1000
        self.assertEqual(source_spans(clause),{'S1':clause})

    def test_qwen_adapter_uses_source_reference_protocol(self):
        import json
        from qwen_judge import QwenJudge
        text='第18话，茱莉亚入职星球日报，并结识克拉克。'
        judge=QwenJudge.__new__(QwenJudge)
        judge.last_generation={}
        def generate(messages,max_new_tokens):
            payload=json.loads(messages[1]['content'])
            self.assertEqual(payload['source_spans']['S3'],'并结识克拉克')
            return json.dumps({'records':[{'k':'relation','e':'茱莉亚','x':'克拉克','c':'结识','s':'S3'}]},ensure_ascii=False)
        judge._generate=generate
        result=judge.extract_world(text,{'names':[],'times':[]})
        self.assertEqual(result['records'][0]['text'],'并结识克拉克')
        self.assertEqual(result['prompt_version'],'world-extraction-v4')
