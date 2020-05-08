import unittest

from numpy import testing

from cameramodels import PinholeCameraModel


class TestPinholeCameraModel(unittest.TestCase):

    def test_calc_f_from_fov(self):
        f = PinholeCameraModel.calc_f_from_fov(90, 480)
        testing.assert_almost_equal(
            f, 240)

    def test_calc_fovx(self):
        fovx = PinholeCameraModel.calc_fovx(53.8, 1080, 1920)
        testing.assert_almost_equal(
            fovx, 84.1, decimal=1)

        fovx = PinholeCameraModel.calc_fovx(45.0, 480, 640)
        testing.assert_almost_equal(
            fovx, 57.8, decimal=1)

    def test_from_fov(self):
        PinholeCameraModel.from_fov(45, 480, 640)

    def test_from_fovx(self):
        PinholeCameraModel.from_fovx(45, 480, 640)

    def test_from_fovy(self):
        PinholeCameraModel.from_fovy(45, 480, 640)
