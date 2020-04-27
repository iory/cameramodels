import unittest

from numpy import testing

from cameramodels import PinholeCameraModel


class TestPinholeCameraModel(unittest.TestCase):

    def test_calc_fov(self):
        fovx = PinholeCameraModel.calc_fovx(53.8, 1080, 1920)
        testing.assert_almost_equal(
            fovx, 84.1, decimal=1)

        fovx = PinholeCameraModel.calc_fovx(45.0, 480, 640)
        testing.assert_almost_equal(
            fovx, 57.8, decimal=1)
