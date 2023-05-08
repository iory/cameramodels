#!/usr/bin/env python

import argparse

from cameramodels import PinholeCameraModel


def main():
    parser = argparse.ArgumentParser(description='Visualize URDF')
    parser.add_argument(
        '-i', '--input-camerainfo', type=str, help='Input camera info',
        required=True)
    parser.add_argument(
        '-o', '--output-camerainfo', type=str, help='Output camera info',
        required=True)
    parser.add_argument(
        '--width', type=int, help='New width size',
        required=True)
    parser.add_argument(
        '--height', type=int, help='New height size',
        required=True)
    args = parser.parse_args()

    cam = PinholeCameraModel.from_yaml_file(
        args.input_camerainfo
    )
    out_cam = cam.crop_resize_camera_info((args.width, args.height))
    out_cam.dump(args.output_camerainfo, False)


if __name__ == '__main__':
    main()
