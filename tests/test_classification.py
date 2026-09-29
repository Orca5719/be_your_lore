import json
import unittest
from classification_protocol import validate_classification

class ClassificationTests(unittest.TestCase):
    def test_entity_category_and_exact_time(self):
        text = '第三话，雷能控制时间。'
        raw = json.dumps({'entity':'雷','category':'能力','event_time':'第三话'})
        result = validate_classification(raw,text,['能力','经历'])
        self.assertEqual(result['entity'],'雷')
        self.assertEqual(result['event_time'],'第三话')
        self.assertEqual(result['status'],'ok')

    def test_no_invented_entities_times_or_categories(self):
        for entity,category,time in [('莉娅','能力',None),('雷','乱造分类',None),('雷','能力','第九话')]:
            with self.assertRaises(ValueError):
                validate_classification(json.dumps({'entity':entity,'category':category,'event_time':time}),'雷能控制时间。',['能力'])

    def test_ambiguous_subject_requests_clarification(self):
        result = validate_classification('{"entity":null,"category":"能力","event_time":null}','他能控制时间。',['能力'])
        self.assertEqual(result['status'],'needs_clarification')
        self.assertEqual(result['missing'],['entity'])

    def test_author_overrides_and_invalid_json(self):
        raw = '{"entity":null,"category":"能力","event_time":null}'
        result = validate_classification(raw,'他能控制时间。',['能力'],overrides={'entity':'雷','category':'自定义能力','event_time':'第九话'})
        self.assertEqual(result['category'],'自定义能力')
        self.assertEqual(result['event_time'],'第九话')
        for value in ['{}','not json','{"entity":[],"category":"能力","event_time":null}']:
            with self.assertRaises(ValueError):
                validate_classification(value,'雷能控制时间。',['能力'])

    def test_pronoun_is_not_a_named_entity(self):
        result = validate_classification('{"entity":"他","category":"能力","event_time":null}','他能控制时间。',['能力'])
        self.assertIsNone(result['entity'])
        self.assertEqual(result['status'],'needs_clarification')
