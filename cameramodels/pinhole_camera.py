import copy
import yaml

import numpy as np

try:
    import open3d
    enable_open3d = True
except ImportError:
    enable_open3d = False


class PinholeCameraModel(object):

    """A Pinhole Camera Model

    more detail, see http://wiki.ros.org/image_pipeline/CameraInfo
    http://docs.ros.org/api/sensor_msgs/html/msg/CameraInfo.html

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
                 stamp=None):
        self._width = image_width
        self._height = image_height
        self._aspect = 1.0 * self.width / self.height
        self.K = K
        self.D = D
        self.R = R
        self.P = P
        self.full_K = None
        self.full_P = None
        self._fovx = 2.0 * np.rad2deg(np.arctan(self.width / (2.0 * self.fx)))
        self._fovy = 2.0 * np.rad2deg(np.arctan(self.height / (2.0 * self.fy)))
        self.binning_x = None
        self.binning_y = None
        self.roi = roi
        self.tf_frame = tf_frame
        self.stamp = stamp

    def calc_f_from_fov(self, fov, length):
        return length / (2.0 * np.tan(fov * np.pi / 360.0))

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
        return (self._fovx, self._fovy)

    @property
    def fovx(self):
        return self._fovx

    @property
    def fovy(self):
        return self._fovy

    def get_camera_matrix(self):
        """Return camera matrix

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

        """
        return self._D

    @D.setter
    def D(self, d):
        self._D = np.array(d, dtype=np.float32)

    @property
    def open3d_intrinsic(self):
        """Return open3d.camera.PinholeCameraIntrinsic instance.

        Returns
        -------
        intrinsic : open3d.camera.PinholeCameraIntrinsic
            open3d PinholeCameraIntrinsic
        """
        if not enable_open3d:
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
    def from_intrinsic_matrix(intrinsic_matrix, height, width):
        """Return PinholeCameraModel from intrinsic_matrix.

        Parameters
        ----------
        intrinsic_matrix : numpy.ndarray
            [3, 3] intrinsic matrix.
        height : int
            height of camera.
        width : int
            width of camera.

        Returns
        -------
        cameramodel : cameramodels.PinholeCameraModel
            camera model
        """
        K = np.array(intrinsic_matrix, dtype=np.float64)
        P = np.zeros((3, 4), dtype=np.float64)
        P[:3, :3] = K.copy()
        return PinholeCameraModel(height, width, K, P)

    @staticmethod
    def from_yaml_file(filename):
        with open(filename, 'r') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        image_width = data['image_width']
        image_height = data['image_height']
        K = data['camera_matrix']['data']
        P = data['projection_matrix']['data']
        R = data['rectification_matrix']['data']
        D = data['distortion_coefficients']['data']
        return PinholeCameraModel(
            image_height, image_width,
            K, P, R, D)

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
            image_height, image_width,
            K, P, R, D,
            roi,
            tf_frame,
            stamp)

    def project_pixel_to_3d_ray(self, uv, normalize=False):
        """Returns the ray vector

        Returns the unit vector which passes from the camera center to
        through rectified pixel (u, v),
        using the camera :math:`P` matrix.
        This is the inverse of :meth:`project3dToPixel`.

        Parameters
        ----------
        uv : numpy.ndarray
            rectified pixel coordinates
        normalize : bool
            if True, return normalized ray vector (unit vector).
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

    def batch_project_pixel_to_3d_ray(self, uv):
        """Returns the ray vectors

        This function is the batch version of
        :meth:`project_pixel_to_3d_ray`.
        Returns the unit vector which passes from the
        camera center to through rectified pixel (u, v),
        using the camera :math:`P` matrix.
        This is the inverse of :meth:`batch_project3d_to_pixel`.

        Parameters
        ----------
        uv : numpy.ndarray
            rectified pixel coordinates
        """
        x = (uv[:, 0] - self.cx) / self.fx
        y = (uv[:, 1] - self.cy) / self.fy
        z = np.ones(len(x))
        return np.vstack([x, y, z]).T

    def project3d_to_pixel(self, point):
        """Returns the rectified pixel coordinates

        Returns the rectified pixel coordinates (u, v) of the 3D point,
        using the camera :math:`P` matrix.
        This is the inverse of :meth:`projectPixelTo3dRay`.

        Parameters
        ----------
        point : numpy.ndarray
            3D point (x, y, z)
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
                                 project_valid_depth_only=False):
        """Return project uv coordinates points

        Returns the rectified pixel coordinates (u, v) of the 3D points
        using the camera :math:`P` matrix.
        This is the inverse of :math:`batch_project_pixel_to_3d_ray`.

        Parameters
        ----------
        points : numpy.ndarray
            batch of xyz point (batch_size, 3)
        project_valid_depth_only : bool
            If True, return uvs which are in frame.

        Returns
        -------
        points : tuple of uv points
            (us, vs)
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
            uv = uv[
                np.logical_and(
                    np.logical_and(0 <= uv[:, 0], uv[:, 0] < self.width),
                    np.logical_and(0 <= uv[:, 1], uv[:, 1] < self.height))]
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
