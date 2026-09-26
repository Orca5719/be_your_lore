import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from agent_pipeline.report import build_report, render_author_report
from agent_pipeline.__main__ import main
from test_agent_pipeline_report import judged


class AuthorReportTests(unittest.TestCase):
    def test_conflict_text_has_story_quote_reason_and_source(self):
        value=judged(verdict='contradiction')
        value['items'][0]['event']['sources']=[{'text':'测试原句。','start':0,'end':5}]
        # Matching upstream event references are shared by this test fixture.
        text=render_author_report(build_report(value))
        self.assertIn('与已有内容矛盾',text)
        self.assertIn('characters.md',text)
        self.assertIn('原文：测试原句。',text)
        self.assertIn('原因：',text)

    def test_missing_rule_is_candidate_not_confirmed_new_canon(self):
        value=judged(verdict='uncertain')
        value['items'][0]['uncertainty_code']='missing_rule'
        report=build_report(value)
        self.assertEqual(len(report['report']['possible_additions']),1)
        self.assertEqual(report['report']['possible_additions'][0]['kind'],'unclassified')
        self.assertFalse(report['report']['possible_additions'][0]['saved'])
        self.assertIn('可能新增',render_author_report(report))
        self.assertIn('不能据此认定',render_author_report(report))

    def test_other_uncertainty_and_nonactual_are_not_additions(self):
        for origin,code in [('model','ambiguous_reference'),('program_nonactual_scope','missing_rule')]:
            value=judged(verdict='uncertain')
            value['items'][0].update(origin=origin,uncertainty_code=code)
            report=build_report(value)
            self.assertEqual(report['report']['possible_additions'],[])
            self.assertIn('需要确认',render_author_report(report))

    def test_default_terminal_is_text_but_file_remains_complete_json(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'judge.json';out=Path(folder)/'out.json'
            path.write_text(json.dumps(judged(verdict='contradiction')),encoding='utf-8')
            with patch('sys.argv',['pipeline','report','--judge-file',str(path),'--compact','--output',str(out)]),contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(),0)
            self.assertIn('与已有内容矛盾',output.getvalue())
            self.assertIn(str(out.resolve()),output.getvalue())
            self.assertIn('judge',json.loads(out.read_text(encoding='utf-8')))
