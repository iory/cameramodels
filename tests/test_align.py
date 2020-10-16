import unittest

import numpy as np

from cameramodels import align_depth_to_rgb
from cameramodels.data import kinect_v2_camera_info
from cameramodels import PinholeCameraModel


class TestAlign(unittest.TestCase):

    cm = None

    @classmethod
    def setUpClass(cls):
        cls.cm = PinholeCameraModel.from_yaml_file(kinect_v2_camera_info())

    def test_align_depth_to_rgb(self):
        cm = self.cm
        zero_depth = np.zeros((cm.height, cm.width), dtype=np.float32)
        align_depth_to_rgb(zero_depth,
                           cm, cm, np.eye(4))

        nan_depth = np.zeros((cm.height, cm.width), dtype=np.float32)
        nan_depth[:] = np.NaN
        align_depth_to_rgb(zero_depth,
                           cm, cm, np.eye(4))
