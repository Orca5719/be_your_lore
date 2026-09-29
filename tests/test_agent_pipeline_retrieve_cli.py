import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from agent_pipeline.__main__ import main
from test_agent_pipeline_retrieval import filtered


class RetrieveCLITests(unittest.TestCase):
    def test_filter_file_skips_qwen_and_empty_selection_skips_bge(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'filter.json';path.write_text(json.dumps(filtered(0)),encoding='utf-8')
            with patch('sys.argv',['pipeline','retrieve','--filter-file',str(path),'--compact']),patch('qwen_judge.QwenJudge') as qwen,patch('encoder.Encoder') as bge,contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(),0)
            qwen.assert_not_called();bge.assert_not_called()
            self.assertEqual(json.loads(output.getvalue())['stage'],'retrieval')

    def test_invalid_top_k_is_checked_before_reading_or_loading(self):
        with patch('sys.argv',['pipeline','retrieve','--filter-file','unused.json','--top-k','0']),contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(),2)
        self.assertIn('top-k',json.loads(output.getvalue())['error'])

    def test_wrong_stage_has_structured_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'filter.json';path.write_text('{"stage":"understanding"}',encoding='utf-8')
            with patch('sys.argv',['pipeline','retrieve','--filter-file',str(path)]),contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(),2)
            self.assertEqual(json.loads(output.getvalue())['status'],'error')
