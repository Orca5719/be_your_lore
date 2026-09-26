import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from agent_pipeline.verify_freeze import verify_directory


class FreezeVerificationTests(unittest.TestCase):
    def make_archive_freeze(self,root):
        directory=root/'benchmark_freezes'/'agent_pipeline_v1';directory.mkdir(parents=True)
        archive=directory/'snapshot.zip'
        content=b'frozen source\n'
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('agent_pipeline/example.py',content)
        manifest={
            'name':'Agent Pipeline Quality Benchmark','state':'frozen','format_version':2,
            'files':{'agent_pipeline/example.py':hashlib.sha256(content).hexdigest()},
            'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
        }
        (directory/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
        return directory

    def test_archive_freeze_is_verified_from_self_contained_snapshot(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);directory=self.make_archive_freeze(root)
            result=verify_directory(root,directory)
            self.assertEqual(result['status'],'ok')
            self.assertEqual(result['files'],1)
            self.assertTrue(result['archive_ok'])

    def test_archive_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);directory=self.make_archive_freeze(root)
            (directory/'snapshot.zip').write_bytes(b'tampered')
            result=verify_directory(root,directory)
            self.assertEqual(result['status'],'error')
            self.assertFalse(result['archive_ok'])

