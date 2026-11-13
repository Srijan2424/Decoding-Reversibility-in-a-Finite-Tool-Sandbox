import runpy,unittest
from pathlib import Path
M=runpy.run_path(str(Path(__file__).with_name('43_main_dataset_rq2.py')),run_name='test_subject')
class TestRQ2(unittest.TestCase):
 def test_parse(self):
  self.assertEqual(M['parse']('{"label":"reversible"}'),'reversible')
  for raw in ['{"label":"reversible","label":"irreversible"}','{"label":"bad"}','not json','{"label":"reversible","extra":0}']:
   with self.assertRaises((ValueError,TypeError)):M['parse'](raw)
 def test_metrics(self):
  rows=[{'truth':c,'stated':c,'probe':c} for c in M['LABELS']]
  self.assertTrue(all(v==1 for v in M['metrics'](rows).values()))
  rows[0]['stated']='invalid'
  self.assertAlmostEqual(M['metrics'](rows)['stated_accuracy'],2/3)
  self.assertAlmostEqual(M['metrics'](rows)['stated_balanced_accuracy'],2/3)
 def test_constant_kappa(self):
  self.assertEqual(M['agreement'](['reversible']*3,list(M['LABELS'])),0)
if __name__=='__main__':unittest.main()
