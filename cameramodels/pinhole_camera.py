from __future__ import division

import copy
import warnings

import numpy as np
import PIL
from PIL import Image
from PIL import ImageDraw
import yaml


try:
    import cv2
    _cv2_available = True
except ImportError:
    _cv2_available = False


def pil_to_cv2_interpolation(interpolation):
    if isinstance(interpolation, str):
        interpolation = interpolation.lower()
        if interpolation == 'nearest':
            cv_interpolation = cv2.INTER_NEAREST
        elif interpolation == 'bilinear':
            cv_interpolation = cv2.INTER_LINEAR
        elif interpolation == 'bicubic':
            cv_interpolation = cv2.INTER_CUBIC
        elif interpolation == 'lanczos':
            cv_interpolation = cv2.INTER_LANCZOS4
        else:
            raise ValueError(
                'Not valid Interpolation. '
                'Valid interpolation methods are '
                'nearest, bilinear, bicubic and lanczos.')
    else:
        if interpolation == PIL.Image.NEAREST:
            cv_interpolation = cv2.INTER_NEAREST
        elif interpolation == PIL.Image.BILINEAR:
            cv_interpolation = cv2.INTER_LINEAR
        elif interpolation == PIL.Image.BICUBIC:
            cv_interpolation = cv2.INTER_CUBIC
        elif interpolation == PIL.Image.LANCZOS:
            cv_interpolation = cv2.INTER_LANCZOS4
        else:
            raise ValueError(
                'Not valid Interpolation. '
                'Valid interpolation methods are '
                'PIL.Image.NEAREST, PIL.Image.BILINEAR, '
                'PIL.Image.BICUBIC and PIL.Image.LANCZOS.')
    return cv_interpolation


def format_mat(x, precision):
    return ("[%s]" % (
        np.array2string(x, precision=precision,
                        suppress_small=True, separator=", ")
            .replace("[", "").replace("]", "").replace("\n", "\n        ")))


def get_valid_roi(height, width, roi):
    y1, x1, y2, x2 = roi
    y1 = max(0, y1)
    x1 = max(0, x1)
    y2 = min(height, y2)
    x2 = min(width, x2)
    roi_height = y2 - y1
    roi_width = x2 - x1

    # ROI with non-positive height or width is
    # considered the same as full resolution.
    if x1 == 0 and y1 == 0 \
       and roi_width == 0 and roi_height == 0:
        return [0, 0, height, width]
    elif roi_width <= 0 or roi_height <= 0:
        warnings.warn(
            "Invalid ROI, [left: {}, top: {}, right: {}, bottom: {}]".format(
                roi[0], roi[1], roi[2], roi[3]))
        return [0, 0, height, width]
    else:
        return [y1, x1, y2, x2]


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
        [top, left, bottom, right] order.
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
                 binning_y=1,
                 target_size=None):
        self._width = int(image_width)
        self._height = int(image_height)
        self._full_width = full_width or self._width
        self._full_height = full_height or self._height
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
        self._binning_x = binning_x
        self._binning_y = binning_y
        self._roi = roi or [0, 0, self._height, self._width]
        self._target_size = target_size
        self.tf_frame = tf_frame
        self.stamp = stamp

        # finally calculate K and P considering ROI and binning.
        self._adjust()

    def _adjust(self):
        """Adjust K and P for binning and ROI

        """
        y1, x1, y2, x2 = self.roi
        K = self.full_K.copy()
        P = self.full_P.copy()
        # Adjust K and P for binning and ROI
        if self._target_size is not None:
            self._binning_x = (x2 - x1) / self._target_size[0]
            self._binning_y = (y2 - y1) / self._target_size[1]
        K[0, 0] /= self._binning_x
        K[1, 1] /= self._binning_y
        K[0, 2] = (K[0, 2] - x1) / self._binning_x
        K[1, 2] = (K[1, 2] - y1) / self._binning_y
        P[0, 0] /= self._binning_x
        P[1, 1] /= self._binning_y
        P[0, 2] = (P[0, 2] - x1) / self._binning_x
        P[1, 2] = (P[1, 2] - y1) / self._binning_y
        self.K = K
        self.P = P
        self._width = x2 - x1
        self._height = y2 - y1
        self._aspect = 1.0 * self.width / self.height
        self._fovx = 2.0 * np.rad2deg(np.arctan(self.width / (2.0 * self.fx)))
        self._fovy = 2.0 * np.rad2deg(np.arctan(self.height / (2.0 * self.fy)))

        self.mapx = np.ndarray(shape=(self.height, self.width, 1),
                               dtype='float32')
        self.mapy = np.ndarray(shape=(self.height, self.width, 1),
                               dtype='float32')
        cv2.initUndistortRectifyMap(
            self.K, self.D, self.R, self.P,
            (self.width, self.height),
            cv2.CV_32FC1, self.mapx, self.mapy)

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
    def Tx(self):
        """Return Tx.

        For monocular cameras, Tx = Ty = Tz = 0.
        For a stereo pair, the fourth column [Tx Ty Tz]' is related to the
        position of the optical center of the second camera in the first
        camera's frame.

        Returns
        -------
        Tx : numpy.float32
        """
        return self.P[0, 3]

    @property
    def Ty(self):
        """Return Ty.

        For monocular cameras, Tx = Ty = Tz = 0.
        For a stereo pair, the fourth column [Tx Ty Tz]' is related to the
        position of the optical center of the second camera in the first
        camera's frame.

        Returns
        -------
        Ty : numpy.float32
        """
        return self.P[1, 3]

    @property
    def Tz(self):
        """Return Tz.

        For monocular cameras, Tx = Ty = Tz = 0.
        For a stereo pair, the fourth column [Tx Ty Tz]' is related to the
        position of the optical center of the second camera in the first
        camera's frame.

        Returns
        -------
        Tz : numpy.float32
        """
        return self.P[2, 3]

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
    def K_inv(self):
        """Inverse of Intrinsic camera matrix for the raw (distorted) images.

        .. math::
            K^{-1} = \\left(
                \\begin{array}{ccc}
                  1 / f_x & 0 & - c_x / f_x \\\\
                  0 & 1 / f_y & - c_y / f_y \\\\
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
        return np.array([
            [1.0 / self.fx, 0, - self.cx / self.fx],
            [0, 1.0 / self.fy, - self.cy / self.fy],
            [0, 0, 1]])

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
    def binning_x(self, binning_x):
        """Setter of binning_x

        Note that this setter internally changes target_size, K and P.

        Parameters
        -----------
        binning_x : float
            decimation value.
        """
        if binning_x <= 0:
            raise ValueError("binning should be greater than 0.")
        self._binning_x = binning_x
        roi_width = self.roi[3] - self.roi[1]
        _, target_height = self._target_size
        target_width = int(roi_width / binning_x)
        self._binning_x = roi_width / target_width
        self._target_size = (target_width, target_height)
        self._adjust()

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
    def binning_y(self, binning_y):
        """Setter of binning_y

        Note that this setter internally changes target_size, K and P.

        Parameters
        -----------
        binning_y : float
            decimation value.
        """
        if binning_y <= 0:
            raise ValueError("binning should be greater than 0.")
        self._binning_y = binning_y
        roi_height = self.roi[2] - self.roi[0]
        target_width, _ = self._target_size
        target_height = int(roi_height / binning_y)
        self._binning_y = roi_height / target_height
        self._target_size = (target_width, target_height)
        self._adjust()

    @property
    def target_size(self):
        """Return target_size

        Returns
        -------
        self._target_size : None or tuple(int)
            (width, height).
            If this value is `None`, target size is not specified.
        """
        return self._target_size

    @target_size.setter
    def target_size(self, target_size):
        """Setter of target_size

        This setter internally changes value of binning_x, binning_y, K and P.

        Parameters
        ----------
        target_size : tuple(int)
            (width, height)
        """
        if len(target_size) != 2:
            raise ValueError('target_size length should be 2')
        roi_height = self.roi[2] - self.roi[0]
        roi_width = self.roi[3] - self.roi[1]
        self._binning_x = roi_width / target_size[0]
        self._binning_y = roi_height / target_size[1]
        self._target_size = target_size
        self._adjust()

    @property
    def roi(self):
        """Return roi

        Returns
        -------
        self._roi : None or list[float]
            [top, left, bottom, right] order.
        """
        return self._roi

    @roi.setter
    def roi(self, roi):
        """Setter of roi.

        Parameters
        ----------
        roi : list[float]
            [top, left, bottom, right] order.
        """
        self._roi = get_valid_roi(self._full_height, self._full_width, roi)
        self._adjust()

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
        binning_x = 1
        binning_y = 1
        if 'image_width' in data:
            # opencv format
            image_width = data['image_width']
            image_height = data['image_height']
            K = np.array(
                data['camera_matrix']['data'],
                dtype=np.float32).reshape(3, 3)
            P = np.array(
                data['projection_matrix']['data'],
                dtype=np.float32).reshape(3, 4)
            R = np.array(
                data['rectification_matrix']['data'],
                dtype=np.float32).reshape(3, 3)
            D = np.array(
                data['distortion_coefficients']['data'],
                dtype=np.float32)

            distortion_model = 'plumb_bob'
            if 'camera_name' in data:
                name = data['camera_name'] or ''
            else:
                name = ''
        elif 'width' in data:
            # ROS yaml format
            image_width = data['width']
            image_height = data['height']
            K = np.array(
                data['K'],
                dtype=np.float32).reshape(3, 3)
            P = np.array(
                data['P'],
                dtype=np.float32).reshape(3, 4)
            R = np.array(
                data['R'],
                dtype=np.float32).reshape(3, 3)
            D = np.array(
                data['D'],
                dtype=np.float32)
            distortion_model = data['distortion_model']
            name = ''
        else:
            raise RuntimeError("Not supported YAML file.")

        if 'binning_x' in data:
            binning_x = 1 if data['binning_x'] == 0 else data['binning_x']
        if 'binning_y' in data:
            binning_y = 1 if data['binning_y'] == 0 else data['binning_y']

        roi_width = image_width
        roi_height = image_height
        if 'roi' in data:
            x_offset = data['roi']['x_offset']
            y_offset = data['roi']['y_offset']
            roi_width = data['roi']['width']
            roi_height = data['roi']['height']
            roi = get_valid_roi(
                image_height, image_width,
                [y_offset,
                 x_offset,
                 y_offset + roi_height,
                 x_offset + roi_width])
            roi_width = roi[3] - roi[1]
            roi_height = roi[2] - roi[0]

        full_K = K.copy()
        full_P = P.copy()
        return PinholeCameraModel(
            roi_height, roi_width,
            K, P, R, D,
            roi=roi,
            distortion_model=distortion_model,
            name=name,
            full_K=full_K,
            full_P=full_P,
            full_width=image_width,
            full_height=image_height,
            binning_x=binning_x,
            binning_y=binning_y)

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

        if camera_info_msg.binning_x == 0:
            binning_x = 1
        else:
            binning_x = camera_info_msg.binning_x

        if camera_info_msg.binning_y == 0:
            binning_y = 1
        else:
            binning_y = camera_info_msg.binning_y

        raw_roi = copy.copy(camera_info_msg.roi)
        roi = get_valid_roi(
            image_height, image_width,
            [raw_roi.y_offset,
             raw_roi.x_offset,
             raw_roi.y_offset + raw_roi.height,
             raw_roi.x_offset + raw_roi.width])
        roi_width = roi[3] - roi[1]
        roi_height = roi[2] - roi[0]

        tf_frame = camera_info_msg.header.frame_id
        stamp = camera_info_msg.header.stamp

        full_K = K.copy()
        full_P = P.copy()
        return PinholeCameraModel(
            roi_height, roi_width,
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

    def rectify_image(self, raw_img,
                      interpolation=PIL.Image.BILINEAR):
        """Rectify input raw image.

        Parameters
        ----------
        raw_img : numpy.ndarray
            raw image.
        interpolation : int
            interpolation method.
            You can specify, PIL.Image.NEAREST, PIL.Image.BILINEAR,
            PIL.Image.BICUBIC and PIL.Image.LANCZOS.

        Returns
        -------
        rectified_img : numpy.ndarray
            rectified image.
        """
        if _cv2_available is False:
            raise RuntimeError('CV2 are not enabled. Currently '
                               'only support cv2 rectification.')
        cv_interpolation = pil_to_cv2_interpolation(interpolation)
        return cv2.remap(raw_img, self.mapx, self.mapy, cv_interpolation)

    def rectify_point(self, uv_raw):
        """Rectify input raw points.

        Parameters
        ----------
        uv_raw : numpy.ndarray or tuple[float] or list[float]
            raw uv points.

        Returns
        -------
        rectified_uv : numpy.ndarray
            rectified point.
        """
        if _cv2_available is False:
            raise RuntimeError('CV2 are not enabled. Currently '
                               'only support cv2 rectification.')
        src_point = np.array(uv_raw, 'f')
        ndim = src_point.ndim
        if ndim == 2:
            n_points = src_point.shape[0]
        else:
            n_points = 1
        src_point = src_point.reshape(n_points, 1, 2)
        dst = cv2.undistortPoints(src_point, self.K, self.D,
                                  R=self.R, P=self.P)
        if ndim == 1:
            return dst[0, 0]
        else:
            return dst.reshape(-1, 2)

    def unrectify_point(self, uv_points):
        """Return distorted points from rectified points.

        Parameters
        ----------
        uv_points : numpy.ndarray or list
            (u, v) point.

        Returns
        -------
        distorted_points : numpy.ndarray or tuple(int)
            distorted points.
        """
        mapx, mapy = self.mapx, self.mapy
        uv_points = np.array(np.array(uv_points, 'f') + 0.5, 'i')
        if uv_points.ndim == 2:
            u = uv_points[:, 0]
            v = uv_points[:, 1]
        else:
            u = uv_points[0]
            v = uv_points[1]
        x = mapx[v, u]
        y = mapy[v, u]
        if uv_points.ndim == 2:
            unrectified_points = np.concatenate(
                [x.reshape(-1, 1), y.reshape(-1, 1)],
                axis=1)
            return np.array(unrectified_points + 0.5, 'i')
        else:
            return int(x), int(y)

    def crop_resize_camera_info(self, target_size, roi=None):
        """Return cropped and resized region's camera model

        Parameters
        ----------
        target_size : list[int]
            [target_width, target_height] order.
        roi : list[float]
            [top, left, bottom, right] order, by default self.roi.

        Returns
        -------
        cameramodel : cameramodels.PinholeCameraModel
            camera model of cropped and resised region.
        """

        if len(target_size) != 2:
            raise ValueError('target_size length should be 2')

        roi = self.roi if roi is None else roi

        roi_height = roi[2] - roi[0]
        roi_width = roi[3] - roi[1]
        if roi_height <= 0 or roi_width <= 0:
            raise ValueError(
                "Invalid ROI, [left: {}, top: {}, right: {}, bottom: {}]".
                format(roi[0], roi[1], roi[2], roi[3]))

        target_size = self._calc_resize_image_size_keeping_aspect_ratio(
            target_size)
        binning_x = roi_width / target_size[0]
        binning_y = roi_height / target_size[1]

        return PinholeCameraModel(
            self.height, self.width,
            self.K, self.P, self.R, self.D,
            roi,
            self.tf_frame,
            self.stamp,
            name=self.name,
            distortion_model=self.distortion_model,
            full_P=self.full_P,
            full_K=self.full_K,
            full_height=self._full_height,
            full_width=self._full_width,
            binning_x=binning_x,
            binning_y=binning_y,
            target_size=target_size)

    def crop_image(self, img, copy=False):
        """Crop input full resolution image considering roi.

        Note that this function will not return resized image.

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

    def crop_resize_image(self, img, interpolation=PIL.Image.BILINEAR,
                          use_cv2=True):
        """Crop and resize input full resolution image.

        Parameters
        ----------
        img : numpy.ndarray
            input image. (H, W, channel)
        interpolation : int
            interpolation method.
            You can specify, PIL.Image.NEAREST, PIL.Image.BILINEAR,
            PIL.Image.BICUBIC and PIL.Image.LANCZOS.

        Returns
        -------
        out : numpy.ndarray
            cropped and resized image.
        """
        y1, x1, y2, x2 = self.roi
        if self._target_size is None:
            raise ValueError('Target size is not specified')

        H, W = img.shape[:2]
        out_W, out_H = self._target_size
        out_shape = (out_H, out_W)
        if img.ndim == 3:
            _, _, C = img.shape
            out_shape += (C,)
        elif img.ndim == 2:
            pass
        else:
            raise ValueError('Input image is not gray or rgb image.')
        if H != self._full_height or W != self._full_width:
            raise ValueError('Input image shape should be ({}, {})'
                             ', given ({}, {})'.format(
                                 self._full_width, self._full_height, W, H))

        cropped_img = img[y1:y2, x1:x2]
        out = np.empty(out_shape, dtype=img.dtype)
        if use_cv2 and _cv2_available:
            cv_interpolation = pil_to_cv2_interpolation(interpolation)
            out[:] = cv2.resize(cropped_img, self._target_size,
                                interpolation=cv_interpolation)
        else:
            pil_img = Image.fromarray(cropped_img)
            out[:] = pil_img.resize(self._target_size, resample=interpolation)
        return out

    def resize_bbox(self, bbox):
        """Resize input full resolution bbox.

        Parameters
        ----------
        bbox : numpy.ndarray or list[float]
            input bbox. Input shape can be (4,) or (N, 4).
            [top, left, bottom, right] order.

        Returns
        -------
        out_bbox : numpy.ndarray
            resized bbox.
        """
        bbox = np.array(bbox, 'f')
        resize_scales = np.array([self._binning_y, self._binning_x,
                                  self._binning_y, self._binning_x], 'f')
        if bbox.ndim == 1:
            out_bbox = bbox / resize_scales
        elif bbox.ndim == 2:
            out_bbox = bbox / resize_scales.reshape(1, 4)
        else:
            raise ValueError("Not valid bboxes")
        return out_bbox

    def resize_point(self, uv_point):
        """Resize input full resolution uv point.

        Parameters
        ----------
        uv_point : numpy.ndarray or list[float]
            input point. Input shape can be (2,) or (N, 2).
            [u, v] order.

        Returns
        -------
        out_point : numpy.ndarray
            resized point.
        """
        uv_point = np.array(uv_point, 'f')
        resize_scales = np.array([self._binning_x, self._binning_y], 'f')
        if uv_point.ndim == 1:
            out_point = uv_point / resize_scales
        elif uv_point.ndim == 2:
            out_point = uv_point / resize_scales.reshape(1, 2)
        else:
            raise ValueError("Not valid points")
        return out_point

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
        # np.matmul(np.linalg.inv(K), uv)
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

    def points_to_depth(self, points, depth_value=0.0):
        """Return depth image from 3D points.

        Parameters
        ----------
        points : numpy.ndarray
            batch of xyz point (batch_size, 3) or (height, width, 3).
        depth_value : float
            default depth value.

        Returns
        -------
        depth : numpy.ndarray
            projected depth image.
        """
        if points.shape == (self.height, self.width, 3):
            points = points.reshape(-1, 3)
        uv, indices = self.batch_project3d_to_pixel(
            points,
            project_valid_depth_only=True,
            return_indices=True)
        # round off
        uv = np.array(uv + 0.5, dtype=np.int32)
        depth = depth_value * np.ones((self.height, self.width), 'f')
        depth.reshape(-1)[self.flatten_uv(uv)] = points[indices][:, 2]
        return depth

    def depth_to_points(self, depth):
        """Convert depth image to point clouds.

        Parameters
        ----------
        depth : numpy.ndarray
            depth image.

        Returns
        -------
        points : numpy.ndarray
            return shape is (width, height, 3).
        """
        uv = self.flattened_pixel_locations_to_uv(
            np.arange(self.width * self.height))
        points = self.batch_project_pixel_to_3d_ray(
            uv, depth=depth)
        return points.reshape(self.height, self.width, 3)

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
               [ 0.41421356,  0.41421356,  1.        ],
               [ 0.41421356, -0.41421356,  1.        ]])
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
             (np.array([0, 0, height, height, 0]) - cy) *
             np.array([0, max_depth, max_depth, max_depth, max_depth]) / fy,
             np.array([0, max_depth, max_depth, max_depth, max_depth])])
        view_frust_pts = np.dot(rotation, view_frust_pts) + np.tile(
            translation.reshape(3, 1), (1, view_frust_pts.shape[1]))
        return view_frust_pts.T

    def in_view_frustum(self, points, max_depth=100.0):
        """Determine if points in the view frustum.

        Parameters
        ----------
        points : numpy.ndarray or list[tuple(float, float)]
            3D point (x, y, z).
        max_depth : float
            max depth of frustsum.

        Returns
        -------
        ret : numpy.ndarray or bool
            bool array. True indicates point in this view frustsum.
        """
        points = np.array(points)
        view_frust_pts = self.get_view_frustum(max_depth=max_depth)
        camera_center = view_frust_pts[0]
        view_frust_pts = view_frust_pts[1:] - camera_center
        n = np.cross(view_frust_pts, np.roll(view_frust_pts, 1, axis=0))
        n = n / np.linalg.norm(n)
        if points.ndim == 1:
            return np.all(np.dot(n, points - camera_center) > 0)
        elif points.ndim == 2:
            return np.all(
                np.dot(n, (points - camera_center).T) > 0, axis=0)

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

    def dump(self, output_filepath, save_original=True):
        """Dump this camera's parameter to yaml file.

        Parameters
        ----------
        output_filepath : str or pathlib.Path
            output path
        save_original : bool
            If `False`, save resized camera info.
        """
        if save_original is True:
            width = self._full_width
            height = self._full_height
            K = self.full_K
            P = self.full_P
            binning_x = self._binning_x
            binning_y = self._binning_y
            x_offset = self.roi[1]
            y_offset = self.roi[0]
            roi_height = self.roi[2] - self.roi[0]
            roi_width = self.roi[3] - self.roi[1]
        else:
            width = int(self._width / self._binning_x)
            height = int(self._height / self._binning_y)
            K = self.K
            P = self.P
            binning_x = 1
            binning_y = 1
            x_offset = 0
            y_offset = 0
            roi_height = 0
            roi_width = 0

        camera_data = "\n".join([
                "image_width: %d" % width,
                "image_height: %d" % height,
                "camera_name: " + self.name,
                "camera_matrix:",
                "  rows: 3",
                "  cols: 3",
                "  data: " + format_mat(
                    np.array(K.reshape(-1), dtype=np.float64), 5),
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
                    np.array(P.reshape(-1), dtype=np.float64), 5),
                "binning_x: %f" % binning_x,
                "binning_y: %f" % binning_y,
                "roi:",
                "  x_offset: %d" % x_offset,
                "  y_offset: %d" % y_offset,
                "  height: %d" % roi_height,
                "  width: %d" % roi_width,
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

    def points_in_roi(self, points):
        """Check if input points are in roi.

        Parameters
        ----------
        points : list[list[float, float]]
            [[x_1, y_1], [x_2, y_2] ..., [x_n, y_n]].

        Returns
        -------
        result list[bool]
            True if the point is in roi, False otherwise.
        """
        result = []
        for point in points:
            if self.roi[1] <= point[0] <= self.roi[3] \
               and self.roi[0] <= point[1] <= self.roi[2]:
                result.append(True)
            else:
                result.append(False)

        return result

    def _calc_resize_image_size_keeping_aspect_ratio(
            self, target_size):
        """Calculate resized image size keeping aspect ratio.

        Parameters
        ----------
        target_size : tuple(int or None)
            values of (width, height).
            If width or height is None,
            calculate it from aspect ratio.

        Returns
        -------
        resized_size : tuple(int)
            resized target size.
        """
        width, height = target_size
        if width is not None and height is not None:
            return (width, height)
        if width is None and height is None:
            raise ValueError('Only width or height should be specified.')
        if width:
            height = width * self._full_height / self._full_width
        else:
            width = height * self._full_width / self._full_height
        height = int(height)
        width = int(width)
        return (width, height)
