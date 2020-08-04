import copy
import os.path as osp
import unittest

import numpy as np
from numpy import testing
from PIL import Image

from cameramodels.data import kinect_v2_camera_info
from cameramodels.data import kinect_v2_image
from cameramodels import PinholeCameraModel

data_dir = osp.join(osp.abspath(osp.dirname(__file__)), 'data')
camera_info_path = osp.join(data_dir, 'camera_info.yaml')
ros_camera_info_path = osp.join(data_dir, 'ros_camera_info.yaml')


class TestPinholeCameraModel(unittest.TestCase):

    cm = None

    @classmethod
    def setUpClass(cls):
        cls.cm = PinholeCameraModel.from_fovy(45, 480, 640)

    def test_crop_camera_info(self):
        self.cm.crop_camera_info(0, 0, 100, 100)

    def test_crop_image(self):
        cropped_cm = self.cm.crop_camera_info(0, 0, 100, 100)
        img = np.zeros((480, 640))
        ret_img = cropped_cm.crop_image(img)
        testing.assert_equal(ret_img.shape, (100, 100))

        with self.assertRaises(ValueError):
            cropped_cm.crop_image(np.zeros(100))

        with self.assertRaises(ValueError):
            cropped_cm.crop_image(np.zeros((100, 100)))

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

    def test_open3d_intrinsic(self):
        cm = self.cm
        cm.open3d_intrinsic

    def test_batch_project3d_to_pixel(self):
        cm = self.cm
        cm.batch_project3d_to_pixel(
            np.array([[10, 0, 1],
                      [0, 0, 1]]))

        _, valid_indices = cm.batch_project3d_to_pixel(
            np.array([[10, 0, 1],
                      [0, 0, 1]]),
            project_valid_depth_only=True,
            return_indices=True)
        testing.assert_equal(valid_indices, np.array([1]))

    def test_flatten_uv(self):
        cm = self.cm
        flatten_uv = cm.flatten_uv(np.array([(1, 0), (100, 1), (100, 2)]))
        testing.assert_equal(flatten_uv, [1, 740, 1380])

    def test_flattened_pixel_locations_to_uv(self):
        cm = self.cm
        flatten_uv = [1, 740, 1380]
        uv = cm.flattened_pixel_locations_to_uv(flatten_uv)
        testing.assert_equal(uv, [(1, 0), (100, 1), (100, 2)])

    def test_from_yaml_file(self):
        PinholeCameraModel.from_yaml_file(camera_info_path)
        PinholeCameraModel.from_yaml_file(ros_camera_info_path)

    def test_draw_roi(self):
        cm = PinholeCameraModel.from_yaml_file(kinect_v2_camera_info())
        img_org = kinect_v2_image()
        img = img_org.copy()
        cm.draw_roi(img, copy=True)
        testing.assert_equal(img, img_org)

        pil_img = Image.fromarray(img_org)
        gray_pil_img = pil_img.convert("L")
        gray_img_org = np.array(gray_pil_img, dtype=np.uint8)
        gray_img = gray_img_org.copy()
        cm.draw_roi(gray_img, copy=False)
        testing.assert_equal(gray_img, gray_img_org)
        cm.draw_roi(gray_img, copy=True)  # ignore copy=True
        testing.assert_equal(gray_img, gray_img_org)

        alpha_pil_img = pil_img.convert("RGBA")
        alpha_img_org = np.array(alpha_pil_img, dtype=np.uint8)
        alpha_img = alpha_img_org.copy()
        cm.draw_roi(alpha_img, copy=True)
        testing.assert_equal(alpha_img, alpha_img_org)

    def test__roi(self):
        cm = PinholeCameraModel.from_yaml_file(kinect_v2_camera_info())
        org_roi = copy.deepcopy(cm.roi)
        org_K = cm.K.copy()
        org_P = cm.P.copy()

        cm.roi = [0, 0, 100, 100]
        testing.assert_almost_equal(org_K[:3, :2], cm.K[:3, :2])
        testing.assert_almost_equal(cm.K[:2, 2], [951.8467, 506.9212],
                                    decimal=3)
        testing.assert_almost_equal(org_P[:3, :2], cm.P[:3, :2])
        testing.assert_almost_equal(cm.P[:2, 2], [951.8467, 506.9212],
                                    decimal=3)
        cm.roi = org_roi
        testing.assert_almost_equal(org_K, cm.K)
        testing.assert_almost_equal(org_P, cm.P)
