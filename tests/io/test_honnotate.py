import os.path as osp
import unittest

from cameramodels.io import read_honnotate


filepath = osp.join(osp.abspath(osp.dirname(__file__)),
                    'data',
                    'honnotate_intrinsics.txt')


class TestHOnnotate(unittest.TestCase):

    def test_read_honnotate(self):
        read_honnotate(filepath)
