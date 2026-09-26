import itertools
import unittest
import numpy as np
from src.metric import entity_f05,evaluate
from src.decide import expected_f
from src.split import partition
from src.normalize import text

class Core(unittest.TestCase):
    def test_official_example_and_empty_cases(self):
        self.assertAlmostEqual(entity_f05({'a','b'},{'a','b','c'}),5/7)
        self.assertEqual(entity_f05(set(),set()),1)
        self.assertEqual(entity_f05(set(),{'x'}),0)
        self.assertEqual(entity_f05({'x'},set()),0)
        self.assertEqual(entity_f05({'x'},{'y'}),0)
    def test_macro_includes_unretrieved_queries(self):
        r=evaluate({'a':set(),'b':{'t'},'c':{'u'}},{'b':{'t'}})
        self.assertAlmostEqual(r['macro_f0.5'],2/3)
        with self.assertRaises(ValueError):evaluate({'a':set()},{},['missing'])
    def test_expected_f_matches_exhaustive_optimum(self):
        p=np.array([.12,.73,.91,.4])
        def utility(selected):
            value=0
            for bits in itertools.product((0,1),repeat=len(p)):
                prob=np.prod([x if b else 1-x for x,b in zip(p,bits)])
                value+=prob*entity_f05({i for i,b in enumerate(bits) if b},selected)
            return value
        selected=set(expected_f(p))
        all_scores=[utility({i for i,b in enumerate(bits) if b}) for bits in itertools.product((0,1),repeat=len(p))]
        self.assertAlmostEqual(utility(selected),max(all_scores))
    def test_unicode_and_group_stability(self):
        self.assertIn('ಕನ್ನಡ',text('ಕನ್ನಡ'))
        self.assertEqual(partition('S1-example'),partition('S1-example'))

if __name__=='__main__':unittest.main()
