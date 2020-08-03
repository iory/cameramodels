import os.path as osp

import numpy as np
from PIL import Image

data_dir = osp.abspath(osp.dirname(__file__))


def kinect_v2_image():
    pil_img = Image.open(osp.join(data_dir, 'kinect_v2', 'hd.jpg'))
    return np.array(pil_img, dtype=np.uint8)[..., ::-1]


def kinect_v2_camera_info():
    return osp.join(data_dir, 'kinect_v2', 'camera_info.yaml')
