import os.path as osp

import numpy as np

from cameramodels.pinhole_camera import PinholeCameraModel


def read_honnotate(filepath):
    filepath = str(filepath)
    if not osp.exists(filepath):
        raise OSError('Input intrinsic file "{}" is not found.'
                      .format(filepath))
    with open(filepath, 'r') as f:
        line = f.readline()
    line = line.strip()
    items = line.split(',')
    for item in items:
        if 'fx' in item:
            fx = float(item.split(':')[1].strip())
        elif 'fy' in item:
            fy = float(item.split(':')[1].strip())
        elif 'ppx' in item:
            cx = float(item.split(':')[1].strip())
        elif 'ppy' in item:
            cy = float(item.split(':')[1].strip())
        elif 'width' in item:
            width = float(item.split(':')[1].strip())
        elif 'height' in item:
            height = float(item.split(':')[1].strip())

    intrinsic_matrix = np.array([[fx, 0, cx],
                                 [0, fy, cy],
                                 [0, 0, 1]], dtype=np.float32)

    return PinholeCameraModel.from_intrinsic_matrix(
        intrinsic_matrix, height, width)
