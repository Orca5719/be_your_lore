import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from index_store import save_index, load_index

class IndexTests(unittest.TestCase):
    def test_round_trip_and_failed_write_keeps_old(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            vectors = np.eye(2,dtype=np.float32)
            metadata = {'config': {'model':'test'},'chunks':[{'id':'a'},{'id':'b'}]}
            save_index(root,vectors,metadata)
            actual, info = load_index(root)
            np.testing.assert_array_equal(actual,vectors)
            self.assertEqual(info['chunks'],metadata['chunks'])
            pointer = (root/'CURRENT').read_text()
            with self.assertRaises(ValueError):
                save_index(root,np.zeros((2,2),dtype=np.float32),metadata)
            self.assertEqual((root/'CURRENT').read_text(),pointer)
            np.testing.assert_array_equal(load_index(root)[0],vectors)

    def test_corruption_and_config_mismatch(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            save_index(root,np.eye(2,dtype=np.float32),{'config':{'model':'a'},'chunks':[{'id':'a'},{'id':'b'}]})
            with self.assertRaises(ValueError):
                load_index(root,expected_config={'model':'b'})
            version = (root/'CURRENT').read_text().strip()
            path = root/'versions'/version/'embeddings.npy'
            path.write_bytes(b'broken')
            with self.assertRaises(ValueError):
                load_index(root)

    def test_metadata_swap_detected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            save_index(root,np.eye(2,dtype=np.float32),{'config':{},'chunks':[{'id':'a'},{'id':'b'}]})
            version = (root/'CURRENT').read_text().strip()
            path = root/'versions'/version/'metadata.json'
            info = json.loads(path.read_text(encoding='utf-8'))
            info['chunks'].reverse()
            path.write_text(json.dumps(info),encoding='utf-8')
            with self.assertRaises(ValueError):
                load_index(root)

if __name__ == '__main__':
    unittest.main()
