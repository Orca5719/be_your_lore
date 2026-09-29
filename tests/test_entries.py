import tempfile
from pathlib import Path
import unittest
from entries import EntryStore

class EntryTests(unittest.TestCase):
    def test_preview_cancel_save_duplicate_and_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = EntryStore(Path(tmp)/'entries.json')
            chunks = [{'id':'c1','text':'雷拥有两颗心脏。','file':'characters.md','start_line':6,'end_line':6,'heading_path':['测试设定','雷','生理结构']}]
            preview = store.preview('雷', '能力', '雷能控制时间。', chunks, event_time=None)
            self.assertEqual(preview['existing'], [])
            self.assertEqual(len(preview['other_categories']['生理结构']), 1)
            self.assertFalse(store.path.exists())
            saved = store.commit(preview)
            self.assertEqual(saved['text'], '雷能控制时间。')
            reloaded = EntryStore(store.path)
            self.assertEqual(len(reloaded.preview('雷','能力','新能力', chunks)['existing']), 1)
            with self.assertRaises(ValueError):
                reloaded.commit(reloaded.preview('雷','能力','雷能控制时间。', chunks))

    def test_preview_stale_rejected_and_timeline_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = EntryStore(Path(tmp)/'entries.json')
            first = store.preview('雷','经历','雷离城。', [], event_time='第三话')
            second = store.preview('雷','经历','雷归来。', [])
            self.assertEqual(store.commit(first)['event_time'], '第三话')
            with self.assertRaises(ValueError):
                store.commit(second)
            output = Path(tmp)/'entries.md'
            store.export(output)
            self.assertIn('第三话', output.read_text(encoding='utf-8'))

    def test_invalid_input_and_corrupt_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'entries.json'
            store = EntryStore(path)
            for entity,category,text in [('', '能力','a'),('雷','能力\n## fake','a'),('雷','能力','')]:
                with self.assertRaises(ValueError):
                    store.preview(entity,category,text, [])
            path.write_text('{}',encoding='utf-8')
            with self.assertRaises(ValueError):
                store.preview('雷','能力','a', [])

    def test_export_does_not_overwrite_manually_written_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = EntryStore(Path(tmp)/'entries.json')
            output = Path(tmp)/'manual.md'
            output.write_text('我的正式设定',encoding='utf-8')
            with self.assertRaises(ValueError):
                store.export(output)
            self.assertEqual(output.read_text(encoding='utf-8'),'我的正式设定')
