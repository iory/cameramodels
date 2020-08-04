from __future__ import division

import copy
import yaml

import numpy as np
from PIL import Image
from PIL import ImageDraw


def format_mat(x, precision):
    return ("[%s]" % (
        np.array2string(x, precision=precision,
                        suppress_small=True, separator=", ")
            .replace("[", "").replace("]", "").replace("\n", "\n        ")))


class PinholeCameraModel(object):

    """A Pinhole Camera Model

    more detail, see http://wiki.ros.org/image_pipeline/CameraInfo
    http://docs.ros.org/api/sensor_msgs/html/msg/CameraInfo.html

    Parameters
    ----------
    image_height : int
        height of camera image.
    image_width : int
        width of camera image.
    K : numpy.ndarray
        3x3 intrinsic matrix.
    P : numpy.ndarray
        3x4 projection matrix
    R : numpy.ndarray
        3x3 rectification matrix.
    D : numpy.ndarray
        distortion.
    roi : None or list[float]
        [left_y, left_x, right_y, right_x] order.
    tf_frame : None or str
        tf frame. This is for ROS compatibility.
    stamp : None
        timestamp. This is for ROS compatibility.
    distortion_model : str
        type of distortion model.
    name : None or str
        name of this camera.
    full_K : numpy.ndarray or None
        original intrinsic matrix of full resolution.
        If `None`, set copy of K.
    full_P : numpy.ndarray or None
        original projection matrix of full resolution.
        If `None`, set copy of P.
    full_height : int or None
        This value is indicating original image height.
    full_width : int or None
        This value is indicating original image width.
    """

    def __init__(self,
                 image_height,
                 image_width,
                 K,
                 P,
                 R=np.eye(3),
                 D=np.zeros(5),
                 roi=None,
                 tf_frame=None,
                 stamp=None,
                 distortion_model='plumb_bob',
                 name='',
                 full_K=None,
                 full_P=None,
                 full_height=None,
                 full_width=None,
                 binning_x=1,
                 binning_y=1):
        self._width = image_width
        self._height = image_height
        self._full_width = full_width or self._width
        self._full_height = full_height or self._height
        self._aspect = 1.0 * self.width / self.height
        self.K = K
        self.D = D
        self.R = R
        self.P = P
        self.distortion_model = distortion_model
        self.name = name
        if full_K is not None:
            self._full_K = full_K
        else:
            self._full_K = self.K.copy()
        if full_P is not None:
            self._full_P = full_P
        else:
            self._full_P = self.P.copy()
        self._fovx = 2.0 * np.rad2deg(np.arctan(self.width / (2.0 * self.fx)))
        self._fovy = 2.0 * np.rad2deg(np.arctan(self.height / (2.0 * self.fy)))
        self.binning_x = binning_x
        self.binning_y = binning_y
        self.roi = roi or [0, 0, self._height, self._width]
        self.tf_frame = tf_frame
        self.stamp = stamp

    @property
    def width(self):
        """Returns image width

        Returns
        -------
        self._width : int
            image width
        """
        return self._width

    @width.setter
    def width(self, width):
        """Setter of image width

        Parameters
        ----------
        width : float
            image width of this camera
        """
        if width <= 0:
            raise ValueError
        self._width = width
        self._fovx = 2.0 * np.rad2deg(np.arctan(self.width / (2.0 * self.fx)))
        self._aspect = 1.0 * self.width / self.height

    @property
    def height(self):
        """Returns image height

        Returns
        -------
        self._height : int
            image height
        """
        return self._height

    @height.setter
    def height(self, height):
        """Setter of image height

        Parameters
        ----------
        height : float
            image height of this camera
        """
        if height <= 0:
            raise ValueError
        self._height = height
        self._fovy = 2.0 * np.rad2deg(np.arctan(self.height / (2.0 * self.fy)))
        self._aspect = 1.0 * self.width / self.height

    @property
    def aspect(self):
        """Return aspect ratio

        Returns
        -------
        self._aspect : float
            ascpect ratio of this camera.
        """
        return self._aspect

    @property
    def cx(self):
        """Returns x center

        Returns
        -------
        cx : numpy.float32

        """
        return self.P[0, 2]

    @property
    def cy(self):
        """Returns y center

        Returns
        -------
        cy : numpy.float32

        """
        return self.P[1, 2]

    @property
    def fx(self):
        """Returns x focal length

        Returns
        -------
        fx : numpy.float32

        """
        return self.P[0, 0]

    @property
    def fy(self):
        """Returns y focal length

        Returns
        -------
        fy : numpy.float32

        """
        return self.P[1, 1]

    @property
    def fov(self):
        """Property of fov.

        Returns
        -------
        fov : tuple(float)
            tuple of (fovx, fovy).
        """
        return (self._fovx, self._fovy)

    @property
    def fovx(self):
        """Property of horizontal fov.

        Returns
        -------
        self._fovx : float
            horizontal fov of this camera.
        """
        return self._fovx

    @property
    def fovy(self):
        """Property of vertical fov.

        Returns
        -------
        self._fovy : float
            vertical fov of this camera.
        """
        return self._fovy

    def get_camera_matrix(self):
        """Return camera matrix

        Returns
        -------
        camera_matrix : numpy.ndarray
            camera matrix from Projection matrix.
        """
        return self.P[:3, :3]

    @property
    def K(self):
        """Intrinsic camera matrix for the raw (distorted) images.

        .. math::
            K = \\left(
                \\begin{array}{ccc}
                  f_x & 0 & c_x \\\\
                  0 & f_y & c_y \\\\
                  0 & 0 & 1
                \\end{array}
            \\right)

        Projects 3D points in the camera coordinate frame to 2D pixel
        coordinates using the focal lengths (fx, fy) and principal point
        (cx, cy).

        Returns
        -------
        self._K : numpy.ndarray
            3x3 intrinsic matrix.
        """
        return self._K

    @K.setter
    def K(self, k):
        self._K = np.array(k, dtype=np.float32).reshape(3, 3)

    @property
    def P(self):
        """Projection camera_matrix

        By convention, this matrix specifies the intrinsic
        (camera) matrix of the processed (rectified) image.

        .. math::
            P = \\left(
                \\begin{array}{cccc}
                  {f_x}' & 0 & {c_x}' & T_x \\\\
                  0 & {f_y}' & {c_y}' & T_y \\\\
                  0 & 0 & 1 & 0
                \\end{array}
            \\right)

        Returns
        -------
        self._P : numpy.ndarray
            4x3 projection matrix.
        """
        return self._P

    @P.setter
    def P(self, p):
        self._P = np.array(p, dtype=np.float32).reshape(3, 4)
        self._fovx = 2.0 * np.rad2deg(np.arctan(self.width / (2.0 * self.fx)))
        self._fovy = 2.0 * np.rad2deg(np.arctan(self.height / (2.0 * self.fy)))

    @property
    def R(self):
        """Rectification matrix (stereo cameras only)

        A rotation matrix aligning the camera coordinate system to the ideal
        stereo image plane so that epipolar lines in both stereo images are
        parallel.

        Returns
        -------
        self._R : numpy.ndarray
            rectification matrix.
        """
        return self._R

    @R.setter
    def R(self, r):
        self._R = np.array(r, dtype=np.float32).reshape(3, 3)

    @property
    def D(self):
        """Property of distortion parameters

        The distortion parameters, size depending on the distortion model.
        For "plumb_bob", the 5 parameters are: (k1, k2, t1, t2, k3).

        Returns
        -------
        self._D : numpy.ndarray
            distortion array.
        """
        return self._D

    @D.setter
    def D(self, d):
        self._D = np.array(d, dtype=np.float32)

    @property
    def full_K(self):
        """Return the original camera matrix for full resolution

        Returns
        -------
        self.full_K : numpy.ndarray
            intrinsic matrix.
        """
        return self._full_K

    @property
    def full_P(self):
        """Return the projection matrix for full resolution

        Returns
        -------
        self.full_P : numpy.ndarray
            projection matrix.
        """
        return self._full_P

    @property
    def binning_x(self):
        """Return number of pixels to decimate to one horizontally.

        Returns
        -------
        self._binning_x : int
            binning x.
        """
        return self._binning_x

    @binning_x.setter
    def binning_x(self, decimation_x):
        """Setter of binning_x

        Parameters
        -----------
        decimation_x : int
            decimation value.
        """
        self._binning_x = max(int(decimation_x), 1)

    @property
    def binning_y(self):
        """Return number of pixels to decimate to one vertically.

        Returns
        -------
        self._binning_y : int
            binning y.
        """
        return self._binning_y

    @binning_y.setter
    def binning_y(self, decimation_y):
        """Setter of binning_y

        Parameters
        -----------
        decimation_y : int
            decimation value.
        """
        self._binning_y = max(int(decimation_y), 1)

    @property
    def roi(self):
        """Return roi

        Returns
        -------
        self._roi : None or list[float]
            [left_y, left_x, right_y, right_x] order.
        """
        return self._roi

    @roi.setter
    def roi(self, roi):
        """Setter of roi.

        Parameters
        ----------
        roi : list[float]
            [left_y, left_x, right_y, right_x] order.
        """
        y1, x1, y2, x2 = roi
        K = self.full_K.copy()
        K[0, 2] = (K[0, 2] - x1)
        K[1, 2] = (K[1, 2] - y1)
        P = self.full_P.copy()
        P[0, 2] = (P[0, 2] - x1)
        P[1, 2] = (P[1, 2] - y1)

        height = y2 - y1
        width = x2 - x1
        self.K = K
        self.P = P
        self._width = width
        self._height = height
        self._roi = roi

    @property
    def open3d_intrinsic(self):
        """Return open3d.camera.PinholeCameraIntrinsic instance.

        Returns
        -------
        intrinsic : open3d.camera.PinholeCameraIntrinsic
            open3d PinholeCameraIntrinsic
        """
        try:
            import open3d
        except ImportError:
            raise RuntimeError(
                "Open3d is not installed. Please install Open3d")
        intrinsic = open3d.camera.PinholeCameraIntrinsic(
            self.width,
            self.height,
            self.fx,
            self.fy,
            self.cx,
            self.cy)
        return intrinsic

    @staticmethod
    def calc_fovx(fovy, height, width):
        """Return fovx from fovy, height and width.

        Parameters
        ----------
        fovy : float
            field of view in degree.
        height : int
            height of camera.
        width : int
            width of camera.

        Returns
        -------
        fovx : float
            calculated fovx.
        """
        aspect = 1.0 * width / height
        fovx = np.rad2deg(2 * np.arctan(
            np.tan(0.5 * np.deg2rad(fovy)) * aspect))
        return fovx

    @staticmethod
    def calc_fovy(fovx, height, width):
        """Return fovy from fovx, height and width.

        Parameters
        ----------
        fovx : float
            horizontal field of view in degree.
        height : int
            height of camera.
        width : int
            width of camera.

        Returns
        -------
        fovy : float
            calculated fovy.
        """
        aspect = 1.0 * width / height
        fovy = np.rad2deg(
            2 * np.arctan(
                np.tan(0.5 * np.deg2rad(fovx)) / aspect))
        return fovy

    @staticmethod
    def calc_f_from_fov(fov, aperture):
        """Return focal length.

        Parameters
        ----------
        fov : float
            field of view in degree.
        aperture : float
            aperture.

        Returns
        -------
        focal_length : float
            calculated focal length.
        """
        return aperture / (2.0 * np.tan(np.deg2rad(fov / 2.0)))

    @staticmethod
    def from_fov(fovy, height, width, **kwargs):
        """Return PinholeCameraModel from fovy.

        Parameters
        ----------
        fovy : float
            vertical field of view in degree.
        height : int
            height of camera.
        width : int
            width of camera.

        Returns
        -------
        cameramodel : cameramodels.PinholeCameraModel
            camera model
        """
        return PinholeCameraModel.from_fovy(fovy, height, width, **kwargs)

    @staticmethod
    def from_fovx(fovx, height, width, **kwargs):
        """Return PinholeCameraModel from fovx.

        Parameters
        ----------
        fovx : float
            horizontal field of view in degree.
        height : int
            height of camera.
        width : int
            width of camera.

        Returns
        -------
        cameramodel : cameramodels.PinholeCameraModel
            camera model
        """
        fovy = PinholeCameraModel.calc_fovy(fovx, height, width)
        fy = PinholeCameraModel.calc_f_from_fov(fovy, height)
        fx = PinholeCameraModel.calc_f_from_fov(fovx, width)
        K = [fx, 0, width / 2.0,
             0, fy, height / 2.0,
             0, 0, 1]
        P = [fx, 0, width / 2.0, 0,
             0, fy, height / 2.0, 0,
             0, 0, 1, 0]
        return PinholeCameraModel(
            image_height=height,
            image_width=width,
            K=K,
            P=P,
            **kwargs)

    @staticmethod
    def from_fovy(fovy, height, width, **kwargs):
        """Return PinholeCameraModel from fovy.

        Parameters
        ----------
        fovy : float
            vertical field of view in degree.
        height : int
            height of camera.
        width : int
            width of camera.

        Returns
        -------
        cameramodel : cameramodels.PinholeCameraModel
            camera model
        """
        fovx = PinholeCameraModel.calc_fovx(fovy, height, width)
        fy = PinholeCameraModel.calc_f_from_fov(fovy, height)
        fx = PinholeCameraModel.calc_f_from_fov(fovx, width)
        K = [fx, 0, width / 2.0,
             0, fy, height / 2.0,
             0, 0, 1]
        P = [fx, 0, width / 2.0, 0,
             0, fy, height / 2.0, 0,
             0, 0, 1, 0]
        return PinholeCameraModel(
            image_height=height,
            image_width=width,
            K=K,
            P=P,
            **kwargs)

    @staticmethod
    def from_open3d_intrinsic(open3d_pinhole_intrinsic):
        """Return PinholeCameraModel from open3d's pinhole camera intrinsic.

        Parameters
        ----------
        open3d_pinhole_intrinsic : open3d.camera.PinholeCameraIntrinsic
            open3d PinholeCameraIntrinsic

        Returns
        -------
        cameramodel : cameramodels.PinholeCameraModel
            camera model
        """
        width = open3d_pinhole_intrinsic.width
        height = open3d_pinhole_intrinsic.height
        K = open3d_pinhole_intrinsic.intrinsic_matrix
        P = np.zeros((3, 4), dtype=np.float64)
        P[:3, :3] = K.copy()
        return PinholeCameraModel(height, width, K, P)

    @staticmethod
    def from_intrinsic_matrix(intrinsic_matrix, height, width,
                              **kwargs):
        """Return PinholeCameraModel from intrinsic_matrix.

        Parameters
        ----------
        intrinsic_matrix : numpy.ndarray
            [3, 3] intrinsic matrix.
        height : int
            height of camera.
        width : int
            width of camera.
        kwargs : dict
            keyword args. These values are passed to
            cameramodels.PinholeCameraModel

        Returns
        -------
        cameramodel : cameramodels.PinholeCameraModel
            camera model
        """
        K = np.array(intrinsic_matrix, dtype=np.float64)
        P = np.zeros((3, 4), dtype=np.float64)
        P[:3, :3] = K.copy()
        return PinholeCameraModel(height, width, K, P,
                                  **kwargs)

    @staticmethod
    def from_yaml_file(filename):
        """Create instance of PinholeCameraModel from yaml file.

        This function is supporting OpenCV calibration program's
        YAML format and sensor_msgs/CameraInfo's YAML format in ROS.

        Parameters
        ----------
        filename : str
            path of yaml file.

        Returns
        -------
        cameramodel : cameramodels.PinholeCameraModel
            camera model
        """
        with open(filename, 'r') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        roi = None
        if 'image_width' in data:
            # opencv format
            image_width = data['image_width']
            image_height = data['image_height']
            K = data['camera_matrix']['data']
            P = data['projection_matrix']['data']
            R = data['rectification_matrix']['data']
            D = data['distortion_coefficients']['data']
            distortion_model = 'plumb_bob'
            if 'camera_name' in data:
                name = data['camera_name']
            else:
                name = ''
        elif 'width' in data:
            # ROS yaml format
            image_width = data['width']
            image_height = data['height']
            K = data['K']
            P = data['P']
            R = data['R']
            D = data['D']

            # ROI all zeros is considered the same as full resolution
            if 'roi' in data:
                x_offset = data['roi']['x_offset']
                y_offset = data['roi']['y_offset']
                roi_width = data['roi']['width']
                roi_height = data['roi']['height']
                if x_offset == 0 \
                   and y_offset == 0 \
                   and roi_width == 0 \
                   and roi_height == 0:
                    roi_width = image_width
                    roi_height = image_height
                roi = [y_offset,
                       x_offset,
                       y_offset + roi_height,
                       x_offset + roi_width]
            distortion_model = data['distortion_model']
            name = ''
        else:
            raise RuntimeError("Not supported YAML file.")
        return PinholeCameraModel(
            image_height, image_width,
            K, P, R, D, roi=roi,
            distortion_model=distortion_model,
            name=name)

    @staticmethod
    def from_camera_info(camera_info_msg):
        """Return PinholeCameraModel from camera_info_msg

        Parameters
        ----------
        camera_info_msg : sensor_msgs.msg.CameraInfo
            message of camera info.

        Returns
        -------
        cameramodel : cameramodels.PinholeCameraModel
            camera model
        """
        K = np.array(camera_info_msg.K, dtype=np.float32).reshape(3, 3)
        if camera_info_msg.D:
            D = np.array(camera_info_msg.D, dtype=np.float32)
        else:
            D = np.zeros(5)
        R = np.array(camera_info_msg.R, dtype=np.float32).reshape(3, 3)
        P = np.array(camera_info_msg.P, dtype=np.float32).reshape(3, 4)
        image_width = camera_info_msg.width
        image_height = camera_info_msg.height

        # Binning refers here to any camera setting which combines rectangular
        #  neighborhoods of pixels into larger "super-pixels." It reduces the
        #  resolution of the output image to
        #  (width / binning_x) x (height / binning_y).
        # The default values binning_x = binning_y = 0 is consider
        binning_x = max(1, camera_info_msg.binning_x)
        binning_y = max(1, camera_info_msg.binning_y)

        raw_roi = copy.copy(camera_info_msg.roi)
        # ROI all zeros is considered the same as full resolution
        if (raw_roi.x_offset == 0 and raw_roi.y_offset == 0 and
                raw_roi.width == 0 and raw_roi.height == 0):
            raw_roi.width = image_width
            raw_roi.height = image_height

        roi = [raw_roi.y_offset,
               raw_roi.x_offset,
               raw_roi.y_offset + raw_roi.height,
               raw_roi.x_offset + raw_roi.width]
        tf_frame = camera_info_msg.header.frame_id
        stamp = camera_info_msg.header.stamp

        full_K = K.copy()
        full_P = P.copy()
        # Adjust K and P for binning and ROI
        K[0, 0] /= binning_x
        K[1, 1] /= binning_y
        K[0, 2] = (K[0, 2] - raw_roi.x_offset) / binning_x
        K[1, 2] = (K[1, 2] - raw_roi.y_offset) / binning_y
        P[0, 0] /= binning_x
        P[1, 1] /= binning_y
        P[0, 2] = (P[0, 2] - raw_roi.x_offset) / binning_x
        P[1, 2] = (P[1, 2] - raw_roi.y_offset) / binning_y
        return PinholeCameraModel(
            raw_roi.height, raw_roi.width,
            K, P, R, D,
            roi,
            tf_frame,
            stamp,
            distortion_model=camera_info_msg.distortion_model,
            full_K=full_K,
            full_P=full_P,
            full_height=image_height,
            full_width=image_width,
            binning_x=binning_x,
            binning_y=binning_y)

    def crop_camera_info(self, x, y, height, width):
        """Return cropped region's camera model

        +----------------------+--
        |                      | |
        |  (x, y)              | |
        |     +-------+        | self._full_height
        |     |  ROI  | height | |
        |     +-------+        | |
        |       width          | |
        +----------------------+--
        |--self._full_width----|

        Parameters
        ----------
        x : int
            Leftmost pixel of the ROI.
            0 if the ROI includes the left edge of the image.
        y : int
            Topmost pixel of the ROI.
            0 if the ROI includes the top edge of the image.
        height : int
            Height of ROI.
        width : int
            Width of ROI.

        Returns
        -------
        cameramodel : cameramodels.PinholeCameraModel
            camera model of cropped region.
        """
        K = self.full_K.copy()
        K[0, 2] = (K[0, 2] - x)
        K[1, 2] = (K[1, 2] - y)
        P = self.full_P.copy()
        P[0, 2] = (P[0, 2] - x)
        P[1, 2] = (P[1, 2] - y)

        roi = [y, x, y + height, x + width]
        return PinholeCameraModel(
            height, width,
            K, P, self.R, self.D,
            roi,
            self.tf_frame,
            self.stamp,
            distortion_model=self.distortion_model,
            full_P=self.full_P,
            full_K=self.full_K,
            full_height=self._full_height,
            full_width=self._full_width)

    def crop_image(self, img, copy=False):
        """Crop input full resolution image considering roi.

        Parameters
        ----------
        img : numpy.ndarray
            input image. (H, W, channel)
        copy : bool
            if `True`, return copy image.

        Returns
        -------
        cropped_img : numpy.ndarray
            cropped image.
        """
        if img.ndim == 3:
            H, W, _ = img.shape
        elif img.ndim == 2:
            H, W = img.shape
        else:
            raise ValueError('Input image is not gray or rgb image.')
        if H != self._full_height or W != self._full_width:
            raise ValueError('Input image shape should be ({}, {})'
                             ', given ({}, {})'.format(
                                 self._full_width, self._full_height, W, H))
        y1, x1, y2, x2 = self.roi
        if copy:
            return img[y1:y2, x1:x2].copy()
        else:
            return img[y1:y2, x1:x2]

    def project_pixel_to_3d_ray(self, uv, normalize=False):
        """Returns the ray vector

        Returns the unit vector which passes from the camera center to
        through rectified pixel (u, v),
        using the camera :math:`P` matrix.
        This is the inverse of :meth:`project3d_to_pixel`.

        Parameters
        ----------
        uv : numpy.ndarray
            rectified pixel coordinates
        normalize : bool
            if True, return normalized ray vector (unit vector).

        Returns
        -------
        ray_vector : tuple(float)
            ray vector.
        """
        x = (uv[0] - self.cx) / self.fx
        y = (uv[1] - self.cy) / self.fy
        z = 1.0
        if normalize:
            norm = np.sqrt(x*x + y*y + 1)
            x /= norm
            y /= norm
            z /= norm
        return (x, y, z)

    def batch_project_pixel_to_3d_ray(self, uv,
                                      depth=None):
        """Returns the ray vectors

        This function is the batch version of
        :meth:`project_pixel_to_3d_ray`.
        Returns the unit vector which passes from the
        camera center to through rectified pixel (u, v),
        using the camera :math:`P` matrix.
        This is the inverse of :meth:`batch_project3d_to_pixel`.
        If depth is specified, return 3d points.

        Parameters
        ----------
        uv : numpy.ndarray
            rectified pixel coordinates
        depth : None or numpy.ndarray
            depth value. If this value is specified,
            Return 3d points.

        Returns
        -------
        ret : numpy.ndarray
            calculated ray vectors or points(depth is given case).
            Shape of (batch_size, 3)
        """
        x = (uv[:, 0] - self.cx) / self.fx
        y = (uv[:, 1] - self.cy) / self.fy
        if depth is not None:
            z = depth.reshape(-1)
            x = x * z
            y = y * z
        else:
            z = np.ones(len(x))
        return np.vstack([x, y, z]).T

    def project3d_to_pixel(self, point):
        """Returns the rectified pixel coordinates

        Returns the rectified pixel coordinates (u, v) of the 3D point,
        using the camera :math:`P` matrix.
        This is the inverse of :meth `project_pixel_to_3d_ray`.

        Parameters
        ----------
        point : numpy.ndarray
            3D point (x, y, z)

        Returns
        -------
        uv : tuple(float)
            uv coordinates. If point is not in range of this camera model,
            return tuple(float('nan'), float('nan')).
        """
        dst = np.matmul(self.P, np.array(
            [point[0], point[1], point[2], 1.0], 'f').reshape(4, 1))
        x = dst[0, 0]
        y = dst[1, 0]
        w = dst[2, 0]
        if w != 0:
            return (x / w, y / w)
        else:
            return (float('nan'), float('nan'))

    def batch_project3d_to_pixel(self, points,
                                 project_valid_depth_only=False,
                                 return_indices=False):
        """Return project uv coordinates points

        Returns the rectified pixel coordinates (u, v) of the 3D points
        using the camera :math:`P` matrix.
        This is the inverse of :meth:`batch_project_pixel_to_3d_ray`.

        Parameters
        ----------
        points : numpy.ndarray
            batch of xyz point (batch_size, 3)
        project_valid_depth_only : bool
            If True, return uvs which are in frame.
        return_indices : bool
            If this value and project_valid_depth_only are True,
            return valid indices.

        Returns
        -------
        points : numpy.ndarray
            shape of (batch_size, 2).
        """
        points = np.array(points, dtype=np.float32)
        n = len(points)
        points = np.concatenate(
            [points, np.ones(n, dtype=np.float32).reshape(n, 1)], axis=1)
        dst = np.matmul(self.P, points.T).T
        x = dst[:, 0]
        y = dst[:, 1]
        w = dst[:, 2]
        uv = np.concatenate(
            [(x / w).reshape(-1, 1), (y / w).reshape(-1, 1)], axis=1)
        if project_valid_depth_only is True:
            valid_indices = np.logical_and(
                np.logical_and(0 <= uv[:, 0], uv[:, 0] < self.width),
                np.logical_and(0 <= uv[:, 1], uv[:, 1] < self.height))
            uv = uv[valid_indices]
            if return_indices is True:
                return uv, np.where(valid_indices)[0]
        return uv

    def get_view_frustum(self, max_depth=1.0,
                         translation=np.zeros(3),
                         rotation=np.eye(3)):
        """Return View Frustsum of this camera model.

        Parameters
        ----------
        max_depth : float
            max depth of frustsum.
        translation : numpy.ndarray
            translation vector
        rotation : numpy.ndarray
            rotation matrix

        Returns
        -------
        view_frust_pts : numpy.ndarray
            view frust points shape of (5, 3).

        Examples
        --------
        >>> from cameramodels import Xtion
        >>> cameramodel = Xtion()
        >>> cameramodel.get_view_frustum(max_depth=1.0)
        array([[ 0.        ,  0.        ,  0.        ],
               [-0.41421356, -0.41421356,  1.        ],
               [-0.41421356,  0.41421356,  1.        ],
               [ 0.41421356, -0.41421356,  1.        ],
               [ 0.41421356,  0.41421356,  1.        ]])
        """
        height = self.height
        width = self.width
        cx = self.cx
        cy = self.cy
        fx = self.fx
        fy = self.fy
        view_frust_pts = np.array(
            [(np.array([0, 0, 0, width, width]) - cx) *
             np.array([0, max_depth, max_depth, max_depth, max_depth]) / fx,
             (np.array([0, 0, height, 0, height]) - cy) *
             np.array([0, max_depth, max_depth, max_depth, max_depth]) / fy,
             np.array([0, max_depth, max_depth, max_depth, max_depth])])
        view_frust_pts = np.dot(rotation, view_frust_pts) + np.tile(
            translation.reshape(3, 1), (1, view_frust_pts.shape[1]))
        return view_frust_pts.T

    def flatten_uv(self, uv, dtype=np.int64):
        """Flattens uv coordinates to single dimensional tensor.

        This is the inverse of :meth:`flattened_pixel_locations_to_uv`.

        Parameters
        ----------
        uv : numpy.ndarray or list[tuple(float, float)]
            A pair of uv pixels. Shape of (batch_size, 2).
            [(u_1, v_1), (u_2, v_2) ..., (u_n, v_n)].
        dtype : type
            data type. default is numpy.int64.

        Returns
        -------
        ret : numpy.ndarray
            Flattened uv tensor of shape (n, ).

        Examples
        --------
        >>> from cameramodels import PinholeCameraModel
        >>> cm = PinholeCameraModel.from_fovy(45, 480, 640)
        >>> cm.flatten_uv(np.array([(1, 0), (100, 1), (100, 2)]))
        array([   1,  740, 1380])
        """
        uv = np.array(uv)
        return np.array(uv[:, 1], dtype=dtype) * self.width \
            + np.array(uv[:, 0], dtype=dtype)

    def flattened_pixel_locations_to_uv(self, flat_pixel_locations):
        """Flattens pixel locations(single dimension tensor) to uv coordinates.

        This is the inverse of :meth:`flatten_uv`.

        Parameters
        ----------
        flat_pixel_locations : numpy.ndarray or list[float]
            Flattened pixel locations.

        Returns
        -------
        ret : numpy.ndarray
            UV coordinates.

        Examples
        --------
        >>> from cameramodels import PinholeCameraModel
        >>> cm = PinholeCameraModel.from_fovy(45, 480, 640)
        >>> flatten_uv = [1, 740, 1380]
        >>> cm.flattened_pixel_locations_to_uv(flatten_uv)
        array([[  1,   0],
               [100,   1],
               [100,   2]])
        """
        flat_pixel_locations = np.array(flat_pixel_locations, dtype=np.int64)
        return np.hstack([
            (flat_pixel_locations % self.width).reshape(-1, 1),
            (flat_pixel_locations.T // self.width).reshape(-1, 1)])

    def dump(self, output_filepath):
        """Dump this camera's parameter to yaml file.

        Parameters
        ----------
        output_filepath : str or pathlib.Path
            output path
        """
        camera_data = "\n".join([
                "image_width: %d" % self._full_width,
                "image_height: %d" % self._full_height,
                "camera_name: " + self.name,
                "camera_matrix:",
                "  rows: 3",
                "  cols: 3",
                "  data: " + format_mat(
                    np.array(self.full_K.reshape(-1), dtype=np.float64), 5),
                "distortion_model: " + self.distortion_model,
                "distortion_coefficients:",
                "  rows: 1",
                "  cols: %d" % len(self.D),
                "  data: [%s]" % ", ".join(
                    "%8f" % x
                    for x in self.D),
                "rectification_matrix:",
                "  rows: 3",
                "  cols: 3",
                "  data: " + format_mat(
                    np.array(self.R.reshape(-1), dtype=np.float64), 8),
                "projection_matrix:",
                "  rows: 3",
                "  cols: 4",
                "  data: " + format_mat(
                    np.array(self.full_P.reshape(-1), dtype=np.float64), 5),
                ""
            ])
        with open(str(output_filepath), 'w') as f:
            f.write(camera_data)

    def draw_roi(self, bgr_img, color=(46, 204, 113),
                 box_width=None, copy=False):
        """Draw Region of Interest

        Parameters
        ----------
        bgr_img : numpy.ndarray
            input image.
        color : tuple(int)
            RGB order color.
        box_width : None or int
            box width. If `None`, automatically set from image size.
        copy : bool
            If `True`, return copy image.
            If input image is gray image, this option will be ignored.

        Returns
        -------
        img : numpy.ndarray
            ROI drawn image.
        """
        if bgr_img.ndim == 2:
            img = bgr_img
        elif bgr_img.ndim == 3:
            if bgr_img.shape[2] == 3:
                img = bgr_img[..., ::-1]
            elif bgr_img.shape[2] == 4:
                img = bgr_img[..., [2, 1, 0, 3]]
            else:
                raise ValueError('Input image is not valid rgb image')
        else:
            raise ValueError('Input image is not gray or rgb image.')

        overlay = Image.new("RGBA", (img.shape[1], img.shape[0]), (0, 0, 0, 0))
        trans_draw = ImageDraw.Draw(overlay)
        y1, x1, y2, x2 = self.roi
        box_width = box_width or max(int(round(max(overlay.size) / 180)), 1)
        trans_draw.rectangle((x1, y1, x2, y2), outline=color + (255,),
                             width=box_width)
        pil_img = Image.fromarray(img)
        mode = pil_img.mode
        pil_img = pil_img.convert("RGBA")
        pil_img = Image.alpha_composite(pil_img, overlay)
        if mode == 'L' or mode == 'RGB':
            pil_img = pil_img.convert('RGB')
        elif mode == 'RGBA':
            pil_img = pil_img.convert('RGBA')
        else:
            raise NotImplementedError

        rgb_to_bgr_indices = [2, 1, 0]
        if mode == 'RGBA':
            rgb_to_bgr_indices += [3]
        if mode == 'L' or copy:
            return np.array(pil_img, dtype=img.dtype)[..., rgb_to_bgr_indices]
        else:
            np_pil_img = np.array(pil_img, dtype=img.dtype)
            bgr_img[:] = np_pil_img[..., rgb_to_bgr_indices]
            return bgr_img
