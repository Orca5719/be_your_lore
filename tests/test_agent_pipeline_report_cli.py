import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from agent_pipeline.__main__ import main
from test_agent_pipeline_report import judged
from test_agent_pipeline_judge import retrieved,answer
from test_agent_pipeline_filtering import FakeLLM,upstream,decision


class ReportCLITests(unittest.TestCase):
    def test_judge_file_report_never_loads_model_and_output_is_full(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'judge.json';out=Path(directory)/'report.json'
            path.write_text(json.dumps(judged(verdict='contradiction')),encoding='utf-8')
            with patch('sys.argv',['pipeline','report','--judge-file',str(path),'--json','--compact','--output',str(out)]),patch('qwen_judge.QwenJudge') as qwen,patch('agent_pipeline.__main__.retrieve_events') as retrieval,contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(),0)
            qwen.assert_not_called();retrieval.assert_not_called()
            self.assertEqual(json.loads(output.getvalue())['summary']['verdict'],'contradiction')
            self.assertIn('overview',json.loads(output.getvalue())['report'])
            self.assertNotIn('events',json.loads(output.getvalue()))
            self.assertNotIn('findings',json.loads(output.getvalue()))
            self.assertNotIn('sources',json.loads(output.getvalue())['report']['contradictions'][0])
            self.assertEqual(json.loads(output.getvalue())['report']['contradictions'][0]['actors'],['雷'])
            self.assertIn('judge',json.loads(out.read_text(encoding='utf-8')))

    def test_partial_judge_file_remains_exit_two(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'judge.json';path.write_text(json.dumps(judged(status='partial')),encoding='utf-8')
            with patch('sys.argv',['pipeline','report','--judge-file',str(path),'--json','--compact']),contextlib.redirect_stdout(io.StringIO()) as output,contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(),2)
            self.assertEqual(json.loads(output.getvalue())['status'],'partial')

    def test_raw_story_shares_llm_and_runs_report_without_extra_generation(self):
        model=FakeLLM([{'decisions':[decision('E1')]},answer()])
        with patch('sys.argv',['pipeline','report','--text','雷有两颗心脏。','--json','--compact']),patch('qwen_judge.QwenJudge',return_value=model) as qwen,patch('agent_pipeline.__main__.understand',return_value=upstream(1)),patch('agent_pipeline.__main__.retrieve_events',return_value=retrieved()),contextlib.redirect_stdout(io.StringIO()) as output,contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(),0)
        qwen.assert_called_once();self.assertEqual(len(model.messages),2)
        self.assertEqual(json.loads(output.getvalue())['stage'],'report')
