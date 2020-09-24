import argparse

import cameramodels
import cv2
import numpy as np
import open3d

from cameramodels.align import align_depth_to_rgb


rotation_matrix = np.array(
    [[1, 0, 0, 0],
     [0, -1, 0, 0],
     [0, 0, -1, 0],
     [0, 0, 0, 1]],
    dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(
        description='Align Example')
    parser.add_argument("--no-rgb-rect", action='store_true')
    parser.add_argument("--no-depth-rect", action='store_true')
    args = parser.parse_args()

    depth_to_rgb_transform = np.load(
        './data/depth_to_rgb_transformation.npy')

    depth_cameramodel = cameramodels.PinholeCameraModel.from_yaml_file(
        './data/k4a_depth_camera_info.yaml')
    bgr_cameramodel = cameramodels.PinholeCameraModel.from_yaml_file(
        './data/k4a_rgb_camera_info.yaml')
    depth_img = cv2.imread('./data/k4a_depth_image_raw.png',
                           cv2.IMREAD_ANYDEPTH)
    bgr_img = cv2.imread('./data/k4a_rgb_image_raw.jpg')

    if args.no_depth_rect is False:
        depth_img = depth_cameramodel.rectify_image(
            depth_img,
            interpolation='nearest')

    if args.no_rgb_rect is False:
        bgr_img = bgr_cameramodel.rectify_image(
            bgr_img,
            interpolation='bilinear')

    aligned_img = align_depth_to_rgb(
        depth_img,
        bgr_cameramodel,
        depth_cameramodel,
        depth_to_rgb_transform)

    rgbd_image = open3d.geometry.RGBDImage.create_from_color_and_depth(
        open3d.geometry.Image(cv2.cvtColor(bgr_img, cv2.COLOR_RGBA2BGR)),
        open3d.geometry.Image(aligned_img),
        convert_rgb_to_intensity=False
    )
    pcd = open3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        bgr_cameramodel.open3d_intrinsic,
    )
    pcd.transform(rotation_matrix)
    open3d.visualization.draw_geometries([pcd])


if __name__ == '__main__':
    main()
