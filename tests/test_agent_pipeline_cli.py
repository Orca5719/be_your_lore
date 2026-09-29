import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from agent_pipeline.__main__ import main


class CLITests(unittest.TestCase):
    def test_existing_output_is_not_overwritten_or_model_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'report.json';path.write_text('keep',encoding='utf-8')
            with patch('sys.argv',['pipeline','understand','--text','雷喝水。','--output',str(path)]),patch('agent_pipeline.__main__.understand') as call,contextlib.redirect_stdout(io.StringIO()) as stream:
                self.assertEqual(main(),2)
            call.assert_not_called()
            self.assertEqual(path.read_text(encoding='utf-8'),'keep')
            self.assertEqual(json.loads(stream.getvalue())['status'],'error')

    def test_compact_stdout_and_complete_file(self):
        report=dict(schema_version='test',stage='understanding',status='ok',events=[{'id':'E1','actors':['雷'],'event':'喝水','sources':[{'text':'雷喝水。'}]}],notice='test',coverage={'missing_source_ids':[],'note':'test'},rejected=[],calls=[{'raw_output':'saved'}])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'report.json'
            with patch('sys.argv',['pipeline','understand','--text','雷喝水。','--compact','--output',str(path)]),patch('agent_pipeline.__main__.understand',return_value=report),contextlib.redirect_stdout(io.StringIO()) as stream,contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(),0)
            self.assertNotIn('calls',json.loads(stream.getvalue()))
            self.assertNotIn('sources',json.loads(stream.getvalue())['events'][0])
            self.assertEqual(json.loads(path.read_text(encoding='utf-8'))['calls'],report['calls'])
            self.assertEqual(json.loads(path.read_text(encoding='utf-8'))['events'][0]['sources'],report['events'][0]['sources'])

    def test_unreadable_input_file_has_machine_readable_error(self):
        with patch('sys.argv',['pipeline','understand','--file','nonexistent_module_test_input.txt']),patch('agent_pipeline.__main__.understand') as call,contextlib.redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(main(),2)
        call.assert_not_called()
        self.assertEqual(json.loads(stream.getvalue())['status'],'error')
