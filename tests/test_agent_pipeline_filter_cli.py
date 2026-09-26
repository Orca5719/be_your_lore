import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from agent_pipeline.__main__ import main
from test_agent_pipeline_filtering import upstream,FakeLLM,decision


class FilterCLITests(unittest.TestCase):
    def test_events_report_does_not_repeat_understanding(self):
        report=upstream(0)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'events.json';path.write_text(json.dumps(report),encoding='utf-8')
            with patch('sys.argv',['pipeline','filter','--events-file',str(path),'--compact']),patch('agent_pipeline.__main__.understand') as understand,contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(),0)
            understand.assert_not_called()
            self.assertEqual(json.loads(output.getvalue())['stage'],'filtering')

    def test_invalid_json_returns_structured_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'events.json';path.write_text('invalid',encoding='utf-8')
            with patch('sys.argv',['pipeline','filter','--events-file',str(path)]),contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(),2)
            self.assertEqual(json.loads(output.getvalue())['status'],'error')

    def test_events_file_is_not_allowed_for_understand(self):
        with patch('sys.argv',['pipeline','understand','--events-file','unused.json']),contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(),2)
        self.assertIn('仅用于filter',json.loads(output.getvalue())['error'])

    def test_raw_story_shares_one_model_instance_between_stages(self):
        model=FakeLLM([{'decisions':[decision('E1')]}])
        with patch('sys.argv',['pipeline','filter','--text','雷喝水。']),patch('qwen_judge.QwenJudge',return_value=model) as loader,patch('agent_pipeline.__main__.understand',return_value=upstream(1)) as understand,contextlib.redirect_stdout(io.StringIO()) as output,contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(),0)
        loader.assert_called_once_with('auto')
        self.assertIs(understand.call_args.kwargs['llm'],model)
        self.assertEqual(json.loads(output.getvalue())['stage'],'filtering')
