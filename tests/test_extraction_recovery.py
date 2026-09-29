import unittest
from world_extraction import decode_records, source_spans

class RecoveryTests(unittest.TestCase):
    def test_missing_time_is_null_only_if_not_literal_input(self):
        text='雷有两颗心脏。'
        row={'k':'attribute','e':'雷','c':'生理','s':'S1','t':'未知','f':''}
        records=decode_records({'records':[row]},text,source_spans(text))
        self.assertIsNone(records[0]['time'])
        self.assertIsNone(records[0]['valid_from'])
        row['t']='2062年'
        with self.assertRaises(ValueError):
            decode_records({'records':[row]},text,source_spans(text))
        text='未知星有雷。'
        row['t']='未知'
        self.assertEqual(decode_records({'records':[row]},text,source_spans(text))[0]['time'],'未知')

    def test_partial_recovery_reports_invalid_and_duplicate_rows(self):
        text='雷加入议会。'
        good={'k':'relation','e':'雷','x':'议会','c':'加入','s':'S1'}
        bad=dict(good,x='银河议会')
        result=decode_records({'records':[good,bad,good]},text,source_spans(text),recover=True)
        self.assertEqual(len(result['records']),1)
        self.assertEqual(result['rejected'][0]['record_index'],2)
        self.assertIn('target',result['rejected'][0]['error'])
        self.assertEqual(result['duplicates'],[3])
        with self.assertRaises(ValueError):
            decode_records({'records':[bad]},text,source_spans(text),recover=True)
        with self.assertRaises(ValueError):
            decode_records({'records':[good]*21},text,source_spans(text),recover=True)

    def test_catalog_separates_time_headings(self):
        from world_records import build_catalog
        result=build_catalog([{'text':'地震。','heading_path':['历史','2063年','地震']},
                              {'text':'告知。','heading_path':['历史','第三话','告知']},
                              {'text':'双心脏。','heading_path':['人物','雷','生理']}],[])
        self.assertEqual(result['names'],['雷'])
        self.assertEqual(result['times'],['2063年','第三话'])

    def test_multiple_views_require_all_verdicts_correct(self):
        from world_benchmark import score_case
        case={'claims':[{'input_quote':'雷加入议会','expected_verdict':'一致','allowed_kinds':['attribute','relation'],'evidence':[]}],'expected_overall':'一致'}
        items=[{'proposed':{'kind':kind,'text':'雷加入议会','entity':'雷'},
                'review':{'status':'ok','verdict':'一致','evidence':[]}} for kind in ['attribute','relation']]
        preview={'status':'pending_confirmation','items':items}
        self.assertEqual(score_case(case,preview)['verdict_correct'],1)
        items[1]['review']['verdict']='矛盾'
        self.assertEqual(score_case(case,preview)['verdict_correct'],0)
        preview['status']='partial_extraction'
        self.assertFalse(score_case(case,preview)['complete'])
