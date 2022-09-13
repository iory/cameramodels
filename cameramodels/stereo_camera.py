import numpy as np
from scipy import linalg

from cameramodels import PinholeCameraModel


def direct_linear_transform(P1, P2, point1, point2):
    A = [point1[1] * P1[2, :] - P1[1, :],
         P1[0, :] - point1[0]*P1[2, :],
         point2[1] * P2[2, :] - P2[1, :],
         P2[0, :] - point2[0]*P2[2, :],
         ]
    A = np.array(A).reshape((4, 4))
    B = np.dot(A.transpose(), A)
    U, s, Vh = linalg.svd(B, full_matrices=False)
    return Vh[3, 0:3] / Vh[3, 3]


class StereoCameraModel(object):

    """A Stereo Camera Model

    Parameters
    ----------
    left : cameramodels.PinholeCameraModel
        left camera model.
    right : cameramodels.PinholeCameraModel
        right camera model.
    """

    def __init__(self, left=None, right=None):
        self._left_camera = left or PinholeCameraModel()
        self._right_camera = right or PinholeCameraModel()
        self.update_q()

    @property
    def left_camera(self):
        """Getter of left camera.

        Returns
        -------
        self._left_camera : cameramodels.PinholeCameraModel
            left camera model
        """
        return self._left_camera

    @property
    def right_camera(self):
        """Getter of right camera.

        Returns
        -------
        self._right_camera : cameramodels.PinholeCameraModel
            right camera model
        """
        return self._right_camera

    @property
    def baseline(self):
        """Return left to right baseline.

        Currently assuming horizontal baseline

        Returns
        -------
        baseline : float
            left to right translation.
        """
        return -self._right_camera.Tx / self._right_camera.fx

    @property
    def Q(self):
        """Return Q matrix.

        .. math::
            Q = \\left(
                \\begin{array}{cccc}
                  1 & 0 & 0 & -c_x \\\\
                  0 & 1 & 0 & -c_y \\\\
                  0 & 0 & 0 & f_x \\\\
                  0 & 0 & -1/T_x & (c_x - c'_{x})/T_x
                \\end{array}
            \\right)

        Returns
        -------
        self._Q : numpy.ndarray
            Q matrix.
        """
        return self._Q

    @staticmethod
    def from_yaml_file(left_yaml, right_yaml):
        """Create instance of StereoCameraModel from yaml file.

        This function is supporting sensor_msgs/CameraInfo's YAML format
        in ROS.

        """
        left = PinholeCameraModel.from_yaml_file(left_yaml)
        right = PinholeCameraModel.from_yaml_file(right_yaml)
        model = StereoCameraModel(left, right)
        return model

    @staticmethod
    def from_camera_info(left_camera_info_msg, right_camera_info_msg):
        """Return StereoCameraModel from camera info msgs.

        Parameters
        ----------
        left_camera_info_msg : sensor_msgs.msg.CameraInfo
            left message of camera info.
        right_camera_info_msg : sensor_msgs.msg.CameraInfo
            right message of camera info.

        Returns
        -------
        cameramodel : cameramodels.StereoCameraModel
            stereo camera model
        """
        left = PinholeCameraModel(left_camera_info_msg)
        right = PinholeCameraModel(right_camera_info_msg)
        return StereoCameraModel(left, right)

    def update_q(self):
        """Update variable fields of reprojection matrix

        From Springer Handbook of Robotics, p. 524:

        .. math::
            P = \\left(
                \\begin{array}{cccc}
                  f_x & 0 & c_x & 0 \\\\
                  0 & f_y & c_y & 0 \\\\
                  0 & 0 & 1 & 0
                \\end{array}
            \\right)

        .. math::
            P' = \\left(
                \\begin{array}{cccc}
                  f_x & 0 & c'_x & f_x T_x \\\\
                  0 & f_y & c_y & 0 \\\\
                  0 & 0 & 1 & 0
                \\end{array}
            \\right)

        where primed parameters are from the left projection matrix,
        unprimed from the right.

        ::

            [u   v 1]^T = P  * [x y z 1]^T
            [(u-d) v 1]^T = P' * [x y z 1]^T

        Combining the two equations above results in the following equation

        ::

            [u v u-d 1]^T = [ Fx   0    Cx   0    ] * [ x y z 1]^T
                            [ 0    Fy   Cy   0    ]
                            [ Fx   0    Cx'  FxTx ]
                            [ 0    0    1    0    ]

        Subtracting the 3rd from from the first and inverting the expression
        results in the following equation.

        ::

           [x y z 1]^T = Q * [u v d 1]^T

        Where Q is defined as

        .. math::
            Q = \\left(
                \\begin{array}{cccc}
                  f_y T_x & 0 & 0 & -f_y c_x T_x \\\\
                  0 & f_x T_x & 0 & -f_x c_y T_x \\\\
                  0 & 0 & 0 & f_x f_y T_x \\\\
                  0 & 0 & - f_y & f_y (c_x - c'_x)
                \\end{array}
            \\right)

        Using the assumption f_x = f_y Q can be simplified to the following.
        But for compatibility with stereo cameras with different focal lengths
        we will use the full Q matrix.

        .. math::
            Q = \\left(
                \\begin{array}{cccc}
                  1 & 0 & 0 & -c_x \\\\
                  0 & 1 & 0 & -c_y \\\\
                  0 & 0 & 0 & f_x \\\\
                  0 & 0 & -1/T_x & (c_x - c'_{x})/T_x
                \\end{array}
            \\right)

        .. math::
            Disparity = x_{left} - x_{right}

        For compatibility with stereo cameras with different focal lengths
        we will use the full Q matrix.
        """
        # The baseline member negates our Tx. Undo this negation.
        Tx = - self.baseline
        left = self._left_camera
        right = self._right_camera
        Q = np.zeros((4, 4), dtype=np.float64)
        Q[0, 0] = left.fy * Tx
        Q[0, 3] = -left.fy * left.cx * Tx
        Q[1, 1] = left.fx * Tx
        Q[1, 3] = -left.fx * left.cy * Tx
        Q[2, 3] = left.fx * left.fy * Tx
        Q[3, 2] = -left.fy
        # zero when disparities are pre-adjusted
        Q[3, 3] = left.fy * (left.cx - right.cx)
        self._Q = Q

    def left_depth_to_right_depth(self, depth):
        """Return right camera depth image from left camera depth image.

        Parameters
        ----------
        depth : numpy.ndarray
            depth image in meters.

        Returns
        -------
        right_depth : numpy.ndarray
            right camera's depth image in meters.
        """
        points = self._right_camera.depth_to_points(depth)
        right_depth = self._right_camera.points_to_depth(points)
        return right_depth

    def points_from_keypoints(self, left_uv, right_uv,
                              return_pixel_error=False):
        """Calculate points from left and right camera's keypoints.

        Parameters
        ----------
        left_uv : numpy.ndarray or tuple
            (x, y) coordinates.
        right_uv : numpy.ndarray or tuple
            (x, y) coordinates.
        return_pixel_error : bool
            if `True`, teturns the distance
            between the reprojected point and the input point.

        Returns
        -------
        points : numpy.ndarray
            (x, y, z) points.
        """
        points = direct_linear_transform(
            self._left_camera.P, self._right_camera.P,
            left_uv, right_uv)
        if return_pixel_error:
            left_x, left_y = self._left_camera.project3d_to_pixel(points)
            dist_left = np.sqrt((left_uv[0] - left_x) ** 2
                                + (left_uv[1] - left_y) ** 2)
            right_x, right_y = self._right_camera.project3d_to_pixel(points)
            dist_right = np.sqrt((right_uv[0] - right_x) ** 2
                                 + (right_uv[1] - right_y) ** 2)
            return points, dist_left, dist_right
        return points
