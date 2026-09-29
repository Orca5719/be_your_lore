"""命令入口检查：帮助和参数错误不应触发模型加载。"""
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]

class CliTests(unittest.TestCase):
    def run_cli(self,*args):
        return subprocess.run([sys.executable,'-X','utf8',str(ROOT/'worldcheck.py'),*args],capture_output=True,text=True,encoding='utf-8',timeout=20)

    def test_help_lists_commands(self):
        result=self.run_cli('--help')
        self.assertEqual(result.returncode,0,result.stderr)
        for command in ['index','search','interactive','inspect','benchmark']:
            self.assertIn(command,result.stdout)

    def test_empty_query_fails_before_loading(self):
        result=self.run_cli('search','   ')
        self.assertEqual(result.returncode,2)
        self.assertIn('查询不能为空',result.stderr)
        self.assertEqual(result.stdout,'')
        self.assertNotIn('Traceback',result.stderr)

    def test_inspect_rejects_empty_text_before_loading(self):
        result=self.run_cli('inspect','   ')
        self.assertEqual(result.returncode,2)
        self.assertIn('检查文本不能为空',result.stderr)
        self.assertEqual(result.stdout,'')

    def test_inspect_index_without_model(self):
        from tempfile import TemporaryDirectory
        import numpy as np
        from index_store import save_index
        with TemporaryDirectory() as directory:
            save_index(directory,np.eye(2,dtype=np.float32),{'config':{'model':'unit-test','dimension':2},'sources':{'a.md':'test'},'chunks':[{'id':'a'},{'id':'b'}]})
            result=self.run_cli('inspect','--index',directory,'--json')
            self.assertEqual(result.returncode,0,result.stderr)
            report=json.loads(result.stdout)
            self.assertEqual(report['shape'],[2,2])
            self.assertEqual(report['config']['model'],'unit-test')
            self.assertEqual(report['source_files'],['a.md'])

    def test_invalid_numeric_arguments(self):
        for args in [('search','雷','--k','0'),('index','--batch-size','-1'),('interactive','--max-tokens','513'),('benchmark','--repeats','0'),('benchmark','--threads','0')]:
            result=self.run_cli(*args)
            self.assertEqual(result.returncode,2,result.stderr)
            self.assertIn('必须',result.stderr)
            self.assertEqual(result.stdout,'')
            self.assertNotIn('Traceback',result.stderr)

if __name__=='__main__':
    unittest.main()

