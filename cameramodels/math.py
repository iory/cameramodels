from __future__ import absolute_import

import math

import numpy as np


# epsilon for testing whether a number is close to zero
_EPS = np.finfo(float).eps * 4.0


def _check_valid_rotation(rotation):
    """Checks that the given rotation matrix is valid."""
    rotation = np.array(rotation)
    if not isinstance(
            rotation,
            np.ndarray) or not np.issubdtype(
            rotation.dtype,
            np.number):
        raise ValueError('Rotation must be specified as numeric numpy array')

    if len(rotation.shape) != 2 or \
       rotation.shape[0] != 3 or rotation.shape[1] != 3:
        raise ValueError('Rotation must be specified as a 3x3 ndarray')

    if np.abs(np.linalg.det(rotation) - 1.0) > 1e-3:
        raise ValueError('Illegal rotation. Must have determinant == 1.0, '
                         'get {}'.format(np.linalg.det(rotation)))
    return rotation


def _check_valid_translation(translation):
    """Checks that the translation vector is valid."""
    if not isinstance(
            translation,
            np.ndarray) or not np.issubdtype(
            translation.dtype,
            np.number):
        raise ValueError(
            'Translation must be specified as numeric numpy array')

    t = translation.squeeze()
    if len(t.shape) != 1 or t.shape[0] != 3:
        raise ValueError(
            'Translation must be specified as a 3-vector, '
            '3x1 ndarray, or 1x3 ndarray')


def wxyz2xyzw(quat):
    """Convert quaternion [w, x, y, z] to [x, y, z, w] order.

    Parameters
    ----------
    quat : list or numpy.ndarray
        quaternion [w, x, y, z]
    Returns
    -------
    quaternion : numpy.ndarray
        quaternion [x, y, z, w]

    Examples
    --------
    >>> from skrobot.coordinates.math import wxyz2xyzw
    >>> wxyz2xyzw([1, 2, 3, 4])
    array([2, 3, 4, 1])
    """
    if isinstance(quat, list):
        quat = np.array(quat)
    return np.roll(quat, -1, axis=quat.ndim - 1)


def xyzw2wxyz(quat):
    """Convert quaternion [x, y, z, w] to [w, x, y, z] order.

    Parameters
    ----------
    quat : list or numpy.ndarray
        quaternion [x, y, z, w]

    Returns
    -------
    quaternion : numpy.ndarray
        quaternion [w, x, y, z]

    Examples
    --------
    >>> from skrobot.coordinates.math import xyzw2wxyz
    >>> xyzw2wxyz([1, 2, 3, 4])
    array([4, 1, 2, 3])
    """
    if isinstance(quat, list):
        quat = np.array(quat)
    return np.roll(quat, 1, axis=quat.ndim - 1)


def matrix2quaternion(m):
    """Returns quaternion of given rotation matrix.

    Parameters
    ----------
    m : list or numpy.ndarray
        3x3 rotation matrix

    Returns
    -------
    quaternion : numpy.ndarray
        quaternion [w, x, y, z] order

    Examples
    --------
    >>> import numpy
    >>> from skrobot.coordinates.math import matrix2quaternion
    >>> matrix2quaternion(np.eye(3))
    array([1., 0., 0., 0.])
    """
    m = np.array(m, dtype=np.float64)
    if m.ndim == 2:
        tr = m[0, 0] + m[1, 1] + m[2, 2]
        if tr > 0:
            S = math.sqrt(tr + 1.0) * 2
            qw = 0.25 * S
            qx = (m[2, 1] - m[1, 2]) / S
            qy = (m[0, 2] - m[2, 0]) / S
            qz = (m[1, 0] - m[0, 1]) / S
        elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
            S = math.sqrt(1. + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            qw = (m[2, 1] - m[1, 2]) / S
            qx = 0.25 * S
            qy = (m[0, 1] + m[1, 0]) / S
            qz = (m[0, 2] + m[2, 0]) / S
        elif m[1, 1] > m[2, 2]:
            S = math.sqrt(1. + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            qw = (m[0, 2] - m[2, 0]) / S
            qx = (m[0, 1] + m[1, 0]) / S
            qy = 0.25 * S
            qz = (m[1, 2] + m[2, 1]) / S
        else:
            S = math.sqrt(1. + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            qw = (m[1, 0] - m[0, 1]) / S
            qx = (m[0, 2] + m[2, 0]) / S
            qy = (m[1, 2] + m[2, 1]) / S
            qz = 0.25 * S
        return np.array([qw, qx, qy, qz])
    elif m.ndim == 3:
        r1, r2, r3 = m[:, 0, :], m[:, 1, :], m[:, 2, :]
        r11, r12, r13 = r1[:, 0], r1[:, 1], r1[:, 2]
        r21, r22, r23 = r2[:, 0], r2[:, 1], r2[:, 2]
        r31, r32, r33 = r3[:, 0], r3[:, 1], r3[:, 2]

        q0 = 0.25 * (r11 + r22 + r33 + 1)
        q1 = 0.25 * (r11 - r22 - r33 + 1)
        q2 = 0.25 * (-r11 + r22 - r33 + 1)
        q3 = 0.25 * (-r11 - r22 + r33 + 1)

        q0[np.where(q0 < 0.0)] = 0.
        q1[np.where(q1 < 0.0)] = 0.
        q2[np.where(q2 < 0.0)] = 0.
        q3[np.where(q3 < 0.0)] = 0.

        q0 = np.sqrt(q0)
        q1 = np.sqrt(q1)
        q2 = np.sqrt(q2)
        q3 = np.sqrt(q3)

        ones = np.ones_like(r11)
        aranges = np.arange(ones.shape[0])
        signs_array = np.array([
            [ones, np.sign(r32 - r23), np.sign(r13 - r31), np.sign(r21 - r12)],
            [np.sign(r32 - r23), ones, np.sign(r21 + r12), np.sign(r13 + r31)],
            [np.sign(r13 - r31), np.sign(r21 + r12), ones, np.sign(r32 + r23)],
            [np.sign(r21 - r12), np.sign(r31 + r13), np.sign(r32 + r23), ones],
        ])

        argmaxes = np.argmax(np.array([q0, q1, q2, q3]), axis=0)
        signs = signs_array[:, argmaxes, aranges]

        res = np.array([q0, q1, q2, q3]).T * signs.T
        resq = res / np.linalg.norm(res, axis=1, keepdims=True)
        return resq
    else:
        raise ValueError(
            'Unsupported rotation matrix shape. '
            'Supports rotation matrices of (N, 3, 3) and (3, 3).')


def quaternion2matrix(q, normalize=False):
    """Returns matrix of given quaternion.

    Parameters
    ----------
    quaternion : list or numpy.ndarray
        quaternion [w, x, y, z] order
    normalize : bool
        if normalize is True, input quaternion is normalized.

    Returns
    -------
    rot : numpy.ndarray
        3x3 rotation matrix

    Examples
    --------
    >>> import numpy
    >>> from skrobot.coordinates.math import quaternion2matrix
    >>> quaternion2matrix([1, 0, 0, 0])
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])
    """
    q = np.array(q)
    if normalize:
        q = quaternion_normalize(q)
    else:
        norm = quaternion_norm(q)
        if not np.allclose(norm, 1.0):
            raise ValueError("quaternion q's norm is not 1")
    if q.ndim == 1:
        q0 = q[0]
        q1 = q[1]
        q2 = q[2]
        q3 = q[3]

        m = np.zeros((3, 3))
        m[0, 0] = q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3
        m[0, 1] = 2 * (q1 * q2 - q0 * q3)
        m[0, 2] = 2 * (q1 * q3 + q0 * q2)

        m[1, 0] = 2 * (q1 * q2 + q0 * q3)
        m[1, 1] = q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3
        m[1, 2] = 2 * (q2 * q3 - q0 * q1)

        m[2, 0] = 2 * (q1 * q3 - q0 * q2)
        m[2, 1] = 2 * (q2 * q3 + q0 * q1)
        m[2, 2] = q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3
    elif q.ndim == 2:
        m = np.zeros((q.shape[0], 3, 3), dtype=np.float64)
        m[:, 0, 0] = q[:, 0] * q[:, 0] + \
            q[:, 1] * q[:, 1] - q[:, 2] * q[:, 2] - q[:, 3] * q[:, 3]
        m[:, 0, 1] = 2 * (q[:, 1] * q[:, 2] - q[:, 0] * q[:, 3])
        m[:, 0, 2] = 2 * (q[:, 1] * q[:, 3] + q[:, 0] * q[:, 2])

        m[:, 1, 0] = 2 * (q[:, 1] * q[:, 2] + q[:, 0] * q[:, 3])
        m[:, 1, 1] = q[:, 0] * q[:, 0] - \
            q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2] - q[:, 3] * q[:, 3]
        m[:, 1, 2] = 2 * (q[:, 2] * q[:, 3] - q[:, 0] * q[:, 1])

        m[:, 2, 0] = 2 * (q[:, 1] * q[:, 3] - q[:, 0] * q[:, 2])
        m[:, 2, 1] = 2 * (q[:, 2] * q[:, 3] + q[:, 0] * q[:, 1])
        m[:, 2, 2] = q[:, 0] * q[:, 0] - \
            q[:, 1] * q[:, 1] - q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3]
    return m


def rodrigues(axis, theta=None):
    """Rodrigues formula.

    See: `Rodrigues' rotation formula - Wikipedia
    <https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula>`_.

    See: `Axis-angle representation - Wikipedia
    <https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation>`_.

    Parameters
    ----------
    axis : numpy.ndarray or list
        [x, y, z] vector.
        You can give axis-angle representation to `axis` if `theta` is None.
    theta: float or None (optional)
        radian. If None is given, calculate theta from axis.

    Returns
    -------
    mat : numpy.ndarray
        3x3 rotation matrix

    Examples
    --------
    >>> import numpy
    >>> from skrobot.coordinates.math import rodrigues
    >>> rodrigues([1, 0, 0], 0)
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])
    >>> rodrigues([1, 1, 1], numpy.pi)
    array([[-0.33333333,  0.66666667,  0.66666667],
           [ 0.66666667, -0.33333333,  0.66666667],
           [ 0.66666667,  0.66666667, -0.33333333]])
    """
    axis = np.array(axis, dtype=np.float64)
    if theta is None:
        theta = np.sqrt(np.sum(axis ** 2))
    a = axis / np.linalg.norm(axis)
    cross_prod = np.array([[0, -a[2], a[1]],
                           [a[2], 0, -a[0]],
                           [-a[1], a[0], 0]])
    ctheta = np.cos(theta)
    stheta = np.sin(theta)
    mat = np.eye(3) + \
        cross_prod * stheta + \
        np.matmul(cross_prod, cross_prod) * (1 - ctheta)
    return mat


def rotation_angle(mat):
    """Inverse Rodrigues formula Convert Rotation-Matirx to Axis-Angle.

    Return theta and axis.
    If given unit matrix, return None.

    Parameters
    ----------
    mat : numpy.ndarray
        rotation matrix, shape (3, 3)

    Returns
    -------
    vec : numpy.ndarray
        The rotation vector of the given rotation matrix.

    Notes
    -----
    This function uses the Rodrigues formula to calculate the rotation vector
    from the given rotation matrix. The returned vector has the same direction
    as the axis of rotation and its magnitude is equal to the rotation angle.
    """
    mat = _check_valid_rotation(mat)
    if np.array_equal(mat, np.eye(3)):
        return np.zeros(3, dtype=np.float64)
    theta = np.arccos((np.trace(mat) - 1) / 2)
    axis = 1.0 / (2 * np.sin(theta)) \
        * np.array([mat[2, 1] - mat[1, 2],
                    mat[0, 2] - mat[2, 0],
                    mat[1, 0] - mat[0, 1]])
    return theta * axis


def calc_delta_rvec(gt_r, estimate_r):
    """Calculate the rotation vector.

    Calculate the rotation vector that represents the relative rotation
    between two given rotation matrices.

    Parameters
    ----------
    gt_r : numpy.ndarray
        The ground truth rotation matrix.
    estimate_r : numpy.ndarray
        The estimated rotation matrix.

    Returns
    -------
    vec : numpy.ndarray
        The rotation vector that represents the relative rotation
        between the given rotation matrices.
    """
    delta_r = np.matmul(estimate_r, gt_r.T)
    vec = rotation_angle(delta_r)
    return vec


def quaternion_norm(q):
    """Return the norm of quaternion.

    Parameters
    ----------
    q : list or numpy.ndarray
        [w, x, y, z] order

    Returns
    -------
    norm_q : float
        quaternion norm of q

    Examples
    --------
    >>> from skrobot.coordinates.math import quaternion_norm
    >>> q = [1, 1, 1, 1]
    >>> quaternion_norm(q)
    2.0
    >>> q = [0, 0.7071067811865476, 0, 0.7071067811865476]
    >>> quaternion_norm(q)
    1.0
    """
    q = np.array(q)
    if q.ndim == 1:
        norm_q = np.sqrt(np.dot(q.T, q))
    elif q.ndim == 2:
        norm_q = np.sqrt(np.sum(q * q, axis=1, keepdims=True))
    else:
        raise ValueError
    return norm_q


def quaternion_normalize(q):
    """Return the normalized quaternion.

    Parameters
    ----------
    q : list or numpy.ndarray
        [w, x, y, z] order

    Returns
    -------
    normalized_q : numpy.ndarray
        normalized quaternion

    Examples
    --------
    >>> from skrobot.coordinates.math import quaternion_normalize
    >>> from skrobot.coordinates.math import quaternion_norm
    >>> q = quaternion_normalize([1, 1, 1, 1])
    >>> quaternion_norm(q)
    1.0
    """
    q = np.array(q)
    normalized_q = q / quaternion_norm(q)
    return normalized_q
