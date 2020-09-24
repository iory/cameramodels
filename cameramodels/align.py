import numpy as np


def align_depth_to_rgb(
        depth,
        bgr_cameramodel,
        depth_cameramodel,
        depth_to_rgb_transform):
    """Align depth image to color image.

    Parameters
    ----------
    depth : numpy.ndarray
        depth image in meter order.
    bgr_cameramodel : cameramodels.PinholeCameraModel
        bgr cameramodel
    depth_cameramodel : cameramodels.PinholeCameraModel
        depth cameramodel
    depth_to_rgb_transform : numpy.ndarray
        4x4 transformation matrix.

    Returns
    -------
    aligned_img : numpy.ndarray
        aligned image.
    """
    if depth.shape[0] != depth_cameramodel.height \
       or depth.shape[1] != depth_cameramodel.width:
        raise ValueError

    depth = depth.copy()

    aligned_img = np.zeros((bgr_cameramodel.height, bgr_cameramodel.width),
                           dtype=np.float32)
    depth[np.isnan(depth)] = 0
    v, u = np.array(np.where(depth))
    uv = np.array([u, v]).T

    rotation = depth_to_rgb_transform[:3, :3]
    translation = depth_to_rgb_transform[:3, 3]

    xyz_depth_frame = depth_cameramodel.batch_project_pixel_to_3d_ray(
        uv, depth=depth[depth > 0])
    xyz_rgb_frame = (np.matmul(
        rotation.T, xyz_depth_frame.T)
                     - np.matmul(
                        rotation.T, translation).reshape(3, -1)).T
    rgb_uv, indices = bgr_cameramodel.batch_project3d_to_pixel(
        xyz_rgb_frame,
        project_valid_depth_only=True,
        return_indices=True)
    aligned_img.reshape(-1)[bgr_cameramodel.flatten_uv(rgb_uv)] = \
        depth[depth > 0][indices]
    return aligned_img
