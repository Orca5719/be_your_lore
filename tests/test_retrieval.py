import unittest
import numpy as np
from retrieval import rank_chunks

class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.vectors=np.array([[0.,1.],[1.,0.],[1.,0.]],dtype=np.float32)
        self.chunks=[{'id':str(i),'text':f'片段{i}','file':'a.md','start_line':i+1,'end_line':i+1,'heading_path':['标题']} for i in range(3)]

    def test_scores_order_ties_and_sources(self):
        query=np.array([1.,0.],dtype=np.float32)
        results=rank_chunks(self.vectors,self.chunks,query,k=5)
        self.assertEqual([r['id'] for r in results],['1','2','0'])
        self.assertEqual([r['score'] for r in results],[1.,1.,0.])
        self.assertEqual(results[0]['start_line'],2)
        self.assertEqual(results[0]['text'],'片段1')
        self.assertEqual(len(rank_chunks(self.vectors,self.chunks,query,k=1)),1)

    def test_invalid_k_dimension_and_nonfinite_query(self):
        for k in [0,-1,True,1.5]:
            with self.assertRaises(ValueError):
                rank_chunks(self.vectors,self.chunks,np.array([1.,0.]),k=k)
        for query in [np.ones(3),np.zeros(2),np.array([np.nan,0.]),np.array([[1.,0.]])]:
            with self.assertRaises(ValueError):
                rank_chunks(self.vectors,self.chunks,query)

if __name__=='__main__':
    unittest.main()
