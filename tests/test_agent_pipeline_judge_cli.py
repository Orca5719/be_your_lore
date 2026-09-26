import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from agent_pipeline.__main__ import main
from test_agent_pipeline_judge import retrieved,answer
from test_agent_pipeline_filtering import FakeLLM,upstream,decision


class JudgeCLITests(unittest.TestCase):
    def test_compact_partial_reveals_upstream_rejected_candidates(self):
        report=retrieved(1,'partial')
        report['filtering']['understanding']={'stage':'understanding','status':'partial','rejected':[{'candidate_id':'W1C2'}], 'failure_reasons':{'unresolved_candidates':[{'candidate_id':'W1C2','error':'主体无依据'}]}}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'retrieval.json';path.write_text(json.dumps(report),encoding='utf-8')
            with patch('sys.argv',['pipeline','judge','--retrieval-file',str(path),'--compact']),patch('qwen_judge.QwenJudge',return_value=FakeLLM([answer()])),contextlib.redirect_stdout(io.StringIO()) as output,contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(),2)
            displayed=json.loads(output.getvalue())
            self.assertEqual(displayed['upstream_rejected_count'],1)
            self.assertEqual(displayed['unresolved_upstream_rejected_count'],1)
            self.assertEqual(displayed['failure_reasons']['upstream_stages']['understanding']['unresolved_candidates'][0]['candidate_id'],'W1C2')

    def test_empty_retrieval_file_skips_all_models(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'retrieval.json';path.write_text(json.dumps(retrieved(0)),encoding='utf-8')
            with patch('sys.argv',['pipeline','judge','--retrieval-file',str(path),'--compact']),patch('qwen_judge.QwenJudge') as qwen,contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(),0)
            qwen.assert_not_called()
            self.assertEqual(json.loads(output.getvalue())['stage'],'judge')

    def test_raw_story_reuses_qwen_for_three_llm_stages(self):
        model=FakeLLM([{'decisions':[decision('E1')]},answer()])
        with patch('sys.argv',['pipeline','judge','--text','雷有两颗心脏。','--compact']),patch('qwen_judge.QwenJudge',return_value=model) as loader,patch('agent_pipeline.__main__.understand',return_value=upstream(1)) as understand,patch('agent_pipeline.__main__.retrieve_events',return_value=retrieved()) as retrieval,contextlib.redirect_stdout(io.StringIO()) as output,contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(),0)
        loader.assert_called_once_with('auto')
        self.assertIs(understand.call_args.kwargs['llm'],model)
        self.assertEqual(json.loads(output.getvalue())['items'][0]['verdict'],'consistent')
