import itertools

import cv2
import numpy as np
import scipy as sp

from cameramodels.math import matrix2quaternion
from cameramodels.math import quaternion2matrix
from cameramodels.math import quaternion_normalize


def triangulate_points_from_2d_points(pt2d, P):
    """Triangulates a 3D point from 2D correspondences and camera matrices.

    Parameters
    ----------
    pt2d : array-like, shape (N, 2)
        List of 2D points in homogeneous coordinates.
    P : array-like, shape (N, 3, 4)
        List of camera projection matrices.

    Returns
    -------
    point : ndarray, shape (4,)
        Triangulated 3D point in homogeneous coordinates.

    Raises
    ------
    AssertionError
        If N != len(P) or N < 2.

    Notes
    -----
    This function implements the linear triangulation algorithm described in
    "Multiple View Geometry in Computer Vision" by Hartley and Zisserman.

    """
    N = len(pt2d)
    assert N == len(P)
    assert N >= 2

    AtA = np.zeros((4, 4))
    x = np.zeros((2, 4))
    for i in range(N):
        x[0, :] = P[i][0, :] - pt2d[i][0] * P[i][2, :]
        x[1, :] = P[i][1, :] - pt2d[i][1] * P[i][2, :]
        AtA += x.T @ x

    _, v = np.linalg.eigh(AtA)
    if np.isclose(v[3, 0], 0):
        return v[:, 0]
    else:
        return v[:, 0] / v[3, 0]


def triangulate(R1, t1, R2, t2, n1, n2):
    """Triangulate 3D points from corresponding 2D points in two views.

    Parameters
    ----------
    R1 : array_like, shape (3, 3)
        Rotation matrix of the first camera.
    t1 : array_like, shape (3,)
        Translation vector of the first camera.
    R2 : array_like, shape (3, 3)
        Rotation matrix of the second camera.
    t2 : array_like, shape (3,)
        Translation vector of the second camera.
    n1 : array_like, shape (N, 3)
        Corresponding 2D points in the first view.
    n2 : array_like, shape (N, 3)
        Corresponding 2D points in the second view.

    Returns
    -------
    array_like, shape (N, 3)
        Triangulated 3D points.
    """
    Xh = cv2.triangulatePoints(
        np.hstack([R1, t1[:, None]]),
        np.hstack([R2, t2[:, None]]),
        n1[:, :2].T,
        n2[:, :2].T,
    )
    Xh /= Xh[3, :]
    return Xh[:3, :].T


def z_count(R, t, Xw_Nx3):
    """Counts the number of points in 3D space with positive z coordinate.

    Counts the number of points in 3D space with positive z coordinate
    after projecting them onto the camera using
    the camera extrinsic parameters R and t.

    Parameters
    ----------
    R: numpy.ndarray
        3x3 rotation matrix representing the camera's orientation
        in the world frame.
    t: numpy.ndarray
        3x1 translation vector representing the camera's position
        in the world frame.
    Xw_Nx3: numpy.ndarray
        N x 3 matrix representing N 3D points in the world frame.

    Returns
    -------
    int
        The number of points with a positive z coordinate after projection.
    """
    X = np.matmul(R, Xw_Nx3.T) + t.reshape((3, 1))
    return np.sum(X[2, :] > 0)


def z_test_w2c(R1, t1, R2, t2, n1, n2):
    """Perform z-test.

    Perform z-test to determine whether a triangulated 3D point is in front of
    the cameras or not.

    Parameters
    ----------
    R1 : array_like, shape (3, 3)
        Rotation matrix of the first camera.
    t1 : array_like, shape (3,)
        Translation vector of the first camera.
    R2 : array_like, shape (3, 3)
        Rotation matrix of the second camera.
    t2 : array_like, shape (3,)
        Translation vector of the second camera.
    n1 : array_like, shape (N, 3)
        Corresponding 2D points in the first view.
    n2 : array_like, shape (N, 3)
        Corresponding 2D points in the second view.

    Returns
    -------
    tuple
        A tuple consisting of the sign of the test result, the number
        of 3D points in front of the cameras when the points are triangulated
        with the original camera poses, and the number of 3D points in front
        of the cameras when the cameras are inverted.
    """
    Xp = triangulate(R1, t1, R2, t2, n1, n2)
    Xn = triangulate(R1, -t1, R2, -t2, n1, n2)
    zp = z_count(R1, t1, Xp) + z_count(R2, t2, Xp)
    zn = z_count(R1, t1, Xn) + z_count(R2, t2, Xn)
    return 1 if zp > zn else -1, zp, zn


def collinearity_w2c(R_w2c, n, idx_v, idx_t, num_v, num_t):
    """Construct a sparse matrix A

    Construct a sparse matrix A that represents
    collinearity constraint equations.

    Parameters
    ----------
    R_w2c : array_like, shape (3, 3)
        Rotation matrix from world to camera coordinate system.
    n : array_like, shape (3,)
        Normal vector of the plane on which the collinearity constraints lie.
    idx_v : array_like, shape (n_v,)
        Indices of visible vertices.
    idx_t : array_like, shape (n_t,)
        Indices of visible triangles.
    num_v : int
        Total number of vertices in the mesh.
    num_t : int
        Total number of triangles in the mesh.

    Returns
    -------
    A : sparse matrix, shape (3, (num_v + num_t) * 3)
        Sparse matrix representing the collinearity constraint equations.
    """
    nmat = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    t0 = num_v * 3
    # A = np.zeros((3, (num_v + num_t)*3))
    A = sp.sparse.lil_matrix((3, (num_v + num_t) * 3), dtype=np.float64)
    A[:, idx_v * 3: idx_v * 3 + 3] = np.matmul(nmat, R_w2c)
    A[:, t0 + idx_t * 3: t0 + idx_t * 3 + 3] = nmat

    return A


def coplanarity_w2c(Ra, Rb, na, nb, idx_t1, idx_t2, num_t):
    """Construct a sparse matrix A

    Construct a sparse matrix A that represents
    coplanarity constraint equations.

    Parameters
    ----------
    Ra : array_like, shape (3, 3)
        Rotation matrix for the first set of triangles.
    Rb : array_like, shape (3, 3)
        Rotation matrix for the second set of triangles.
    na : array_like, shape (n, 3)
        Normal vectors of the first set of triangles.
    nb : array_like, shape (n, 3)
        Normal vectors of the second set of triangles.
    idx_t1 : array_like, shape (n_t1,)
        Indices of the first set of triangles.
    idx_t2 : array_like, shape (n_t2,)
        Indices of the second set of triangles.
    num_t : int
        Total number of triangles in the mesh.

    Returns
    -------
    A : sparse matrix, shape (n, num_t * 3)
        Sparse matrix representing the coplanarity constraint equations.
    """
    rows = na.shape[0]
    assert na.shape[1] == 3
    assert nb.shape[0] == rows
    assert nb.shape[1] == 3

    m = np.cross(np.matmul(na, Ra), np.matmul(nb, Rb))
    A = sp.sparse.lil_matrix((rows, num_t * 3), dtype=np.float64)
    A[:, idx_t1 * 3: idx_t1 * 3 + 3] = np.matmul(m, Ra.T)
    A[:, idx_t2 * 3: idx_t2 * 3 + 3] = -np.matmul(m, Rb.T)
    return A


def calib_linear(v_CxNx3, n_CxMx3, compute_x=False):
    """Extrinsic camera calibration.

    Performs linear camera calibration using the collinearity
    and coplanarity constraints.
    See Extrinsic Camera Calibration From a Moving Person
    https://vision.ist.i.kyoto-u.ac.jp/research/calibperson/

    Parameters:
    -----------
    v_CxNx3 : ndarray of shape (C, N, 3)
        Coordinates of 3D points in the world coordinate system.
    n_CxMx3 : ndarray of shape (C, M, 3)
        Coordinates of corresponding 2D image points
        in the camera coordinate system.
    compute_x : bool, optional
        Whether to compute the 3D coordinates of the points. Defaults to False.

    Returns:
    --------
    R_w2c_list : ndarray of shape (C, 3, 3)
        Rotation matrices from world to camera coordinate system.
    t_w2c_list : ndarray of shape (C, 3, 1)
        Translation vectors from world to camera coordinate system.
    scale : float
        Global scale factor.
    X2 : ndarray of shape (M, 3), optional
        If compute_x is True, returns the 3D coordinates of the points
        in the world coordinate system.
    """
    C = v_CxNx3.shape[0]
    v_CxNx3.shape[1]
    M = n_CxMx3.shape[1]
    assert v_CxNx3.shape[2] == 3
    assert n_CxMx3.shape[0] == C
    assert n_CxMx3.shape[2] == 3

    # Rotation
    v_Nx3C = np.hstack(v_CxNx3)
    Y, D, Zt = np.linalg.svd(v_Nx3C)
    # V = Y[:, :3] @ np.diag(D[:3]) / np.sqrt(C)
    R_all = np.sqrt(C) * Zt[:3, :]
    # make R0 be I (also correct handedness)
    Rx = np.linalg.inv(R_all[:3, :3])
    R_all = Rx @ R_all
    assert np.linalg.det(R_all[:3, :3]) > 0

    R_w2c_list = R_all.T.reshape((-1, 3, 3))
    for i, R in enumerate(R_w2c_list):
        u, s, vt = np.linalg.svd(R)
        R = u @ vt
        R_w2c_list[i] = R
    R_w2c_list = quaternion2matrix(quaternion_normalize(
        matrix2quaternion(R_w2c_list)))

    # Translation
    A = []
    for idx_t, (R, n) in enumerate(zip(R_w2c_list, n_CxMx3)):
        for idx_v in range(n.shape[0]):
            A.append(collinearity_w2c(R, n[idx_v, :], idx_v, idx_t, M, C))
    A = sp.sparse.vstack(A)

    B = []
    for ((a, Ra, na), (b, Rb, nb)) in itertools.combinations(
        zip(range(C), R_w2c_list, n_CxMx3), 2
    ):
        B.append(coplanarity_w2c(Ra, Rb, na, nb, a, b, C))
    B = sp.sparse.vstack(B)

    C = sp.sparse.lil_matrix(
        (A.shape[0] + B.shape[0], A.shape[1]), dtype=np.float64)
    C[: A.shape[0]] = A
    C[A.shape[0]:, -B.shape[1]:] = B

    w, v = sp.linalg.eigh(
        (C.T @ C).toarray(), subset_by_index=(0, 5),
        overwrite_a=True, overwrite_b=True
    )
    if w[3] / w[4] > 1e-4:
        # print(f"WARN: degenerate case (only 4 eigenvalues "
        #       "should be zero): lambda={w}")
        pass

    # null-space has 4-dim = any-translation for x/y/z + global-scale
    k = v[:, :4]

    # find a set of coeffs to make t0 be (0, 0, 0)
    _, s, vt = np.linalg.svd(k[-B.shape[1]: -B.shape[1] + 3, :])  # t0
    t = k @ vt[3, :].T  # vt[3] is the coeffs to make t0 zero
    X = t[: -B.shape[1]].reshape((-1, 3))
    t = t[-B.shape[1]:]
    scale = np.linalg.norm(t[3:6])
    t = t / scale
    X = X / scale
    t_w2c_list = t.reshape((-1, 3))

    # z-test to fix the global sign ambiguity
    R1 = R_w2c_list[0]
    R2 = R_w2c_list[1]
    t1 = t_w2c_list[0]
    t2 = t_w2c_list[1]
    n1 = n_CxMx3[0]
    n2 = n_CxMx3[1]
    sign, Np, Nn = z_test_w2c(R1, t1, R2, t2, n1, n2)

    t_w2c_list = sign * t_w2c_list
    X = sign * X

    # recompute X
    if compute_x:
        P = np.concatenate((R_w2c_list, t_w2c_list[:, :, None]), axis=2)
        X2 = []
        for i in range(M):
            x = triangulate_points_from_2d_points(n_CxMx3[:, i, :], P)
            X2.append(x)
        X2 = np.array(X2)[:, :3]
        return R_w2c_list, t_w2c_list.reshape((-1, 3, 1)), scale, X2
    return R_w2c_list, t_w2c_list.reshape((-1, 3, 1)), scale
