import unittest
from hmm import HMM

class HmmInitTest(TestCase):

    def test_load(self):
        self.assertRaises(Exception, HMM())



if __name__ == '__main__':
    unittest.main()