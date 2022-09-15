import copy
import os.path as osp
import unittest

import numpy as np
from numpy import testing
from PIL import Image

import cameramodels
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

    def test_K_inv(self):
        cm = PinholeCameraModel.from_yaml_file(kinect_v2_camera_info())
        testing.assert_almost_equal(
            cm.K_inv, np.linalg.inv(cm.K))

    def test_rectify_image(self):
        cm = PinholeCameraModel.from_yaml_file(kinect_v2_camera_info())
        img = kinect_v2_image()
        cm.rectify_image(img)

    def test_rectify_point(self):
        cm = PinholeCameraModel.from_yaml_file(kinect_v2_camera_info())
        testing.assert_equal(cm.rectify_point([0, 0]).shape, (2,))
        testing.assert_equal(cm.rectify_point((0, 0)).shape, (2,))
        testing.assert_equal(cm.rectify_point(np.array((0, 0))).shape, (2,))

        testing.assert_equal(cm.rectify_point([[0, 0]]).shape, (1, 2))
        testing.assert_equal(
            cm.rectify_point(np.array([[0, 0]])).shape, (1, 2))

        testing.assert_equal(
            cm.rectify_point(np.zeros((10, 2))).shape, (10, 2))

    def test_unrectify_point(self):
        cm = PinholeCameraModel.from_yaml_file(kinect_v2_camera_info())
        testing.assert_equal(len(cm.unrectify_point([0, 0])), 2)
        testing.assert_equal(cm.unrectify_point([[0, 0]]).shape, (1, 2))

    def test_crop_resize_camra_info(self):
        cropped_resized_cm = self.cm.crop_resize_camera_info(
            target_size=[100, 200], roi=[0, 0, 100, 150])
        testing.assert_equal(cropped_resized_cm.binning_x, 150 / 100.)
        testing.assert_equal(cropped_resized_cm.binning_y, 100 / 200.)

        with self.assertRaises(ValueError):
            # invalid roi
            cropped_resized_cm = self.cm.crop_resize_camera_info(
                target_size=[200, 100], roi=[200, 0, 100, 150])

    def test_crop_image(self):
        cropped_cm = copy.deepcopy(self.cm)
        cropped_cm.roi = [0, 0, 100, 101]
        img = np.zeros((480, 640))
        ret_img = cropped_cm.crop_image(img)
        testing.assert_equal(ret_img.shape, (100, 101))

        with self.assertRaises(ValueError):
            cropped_cm.crop_image(np.zeros(100))

        with self.assertRaises(ValueError):
            cropped_cm.crop_image(np.zeros((100, 100)))

    def test_crop_resize_image(self):
        for use_cv2 in [False, True]:
            cropped_cm = copy.deepcopy(self.cm)
            cropped_cm.roi = [0, 0, 100, 100]
            gray_img = np.zeros((480, 640), dtype=np.uint8)
            cropped_cm.target_size = (256, 257)
            ret_img = cropped_cm.crop_resize_image(gray_img,
                                                   use_cv2=use_cv2)
            testing.assert_equal(ret_img.shape, (257, 256))

            bgr_img = np.zeros((480, 640, 3), dtype=np.uint8)
            cropped_cm.target_size = (11, 10)
            ret_img = cropped_cm.crop_resize_image(bgr_img,
                                                   use_cv2=use_cv2)
            testing.assert_equal(ret_img.shape, (10, 11, 3))

            with self.assertRaises(ValueError):
                cropped_cm.crop_resize_image(np.zeros(100),
                                             use_cv2=use_cv2)

            with self.assertRaises(ValueError):
                cropped_cm.crop_resize_image(
                    np.zeros((100, 100), dtype=np.uint8),
                    use_cv2=use_cv2)

    def test_resize_bbox(self):
        resized_cm = self.cm.crop_resize_camera_info(
            target_size=[100, 200])
        out_bbox = resized_cm.resize_bbox(
            [[0, 0, self.cm.height, self.cm.width]])
        testing.assert_almost_equal(
            out_bbox, [[0, 0, 200, 100]], decimal=4)

        out_bbox = resized_cm.resize_bbox(
            [0, 0, self.cm.height, self.cm.width])
        testing.assert_almost_equal(
            out_bbox, [0, 0, 200, 100], decimal=4)

    def test_resize_point(self):
        resized_cm = self.cm.crop_resize_camera_info(
            target_size=[100, 200])
        out_point = resized_cm.resize_point(
            [[0, 0],
             [0, self.cm.height],
             [self.cm.width, self.cm.height],
             [self.cm.width, 0]])
        testing.assert_almost_equal(
            out_point,
            [[0, 0],
             [0, 200],
             [100, 200],
             [100, 0]], decimal=4)

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

    def test_points_roi(self):
        cm = PinholeCameraModel.from_yaml_file(kinect_v2_camera_info())
        points = [[874.5, 680],
                  [875, 680],
                  [875, 679.5],
                  [1072, 680],
                  [1072.5, 680],
                  [1072, 859],
                  [1072, 869.5]]
        testing.assert_equal(
            cm.points_in_roi(points),
            [False, True, False, True, False, True, False])

    def test__target_size(self):
        cm = PinholeCameraModel.from_yaml_file(kinect_v2_camera_info())
        org_roi = copy.deepcopy(cm.roi)
        cm.target_size = (640, 480)
        self.assertEqual(cm.target_size, (640, 480))

        resize_K = cm.K.copy()
        resize_P = cm.P.copy()

        cm.roi = [0, 0, 100, 100]
        cm.target_size = (640, 480)
        cm.roi = org_roi
        testing.assert_equal(resize_K, cm.K)
        testing.assert_equal(resize_P, cm.P)

    def test__binning_x(self):
        cm = PinholeCameraModel.from_yaml_file(kinect_v2_camera_info())
        cm.roi = [0, 0, 100, 100]
        cm.target_size = (640, 480)
        cm.binning_x = 0.5
        self.assertEqual(cm.target_size, (200, 480))

        cm.target_size = (640, 480)
        cm.roi = [0, 0, 100, 97]
        cm.binning_x = 2
        self.assertEqual(cm.target_size, (48, 480))

    def test__binning_y(self):
        cm = PinholeCameraModel.from_yaml_file(kinect_v2_camera_info())
        cm.roi = [0, 0, 100, 100]
        cm.target_size = (640, 480)
        cm.binning_y = 0.5
        self.assertEqual(cm.target_size, (640, 200))

        cm.target_size = (640, 480)
        cm.roi = [0, 0, 97, 100]
        cm.binning_y = 2
        self.assertEqual(cm.target_size, (640, 48))

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

    def test_in_view_frustum(self):
        cm = cameramodels.models.D415()
        testing.assert_equal(cm.in_view_frustum([0, 0, 0.1]), True)
        testing.assert_equal(cm.in_view_frustum([0, 0, 1]), True)
        testing.assert_equal(cm.in_view_frustum([100, 100, 0]), False)
        testing.assert_equal(cm.in_view_frustum([5, 5, 0]), False)

    def test_points_to_depth(self):
        cm = cameramodels.models.AzureKinect()
        depth = np.ones((cm.height, cm.width))
        points = cm.depth_to_points(depth)
        cm.points_to_depth(points)
        cm.points_to_depth(points.reshape(-1, 3))

    def test_depth_to_points(self):
        cm = cameramodels.models.AzureKinect()
        depth = np.ones((cm.height, cm.width))
        cm.depth_to_points(depth)
