from __future__ import division

from cameramodels import pinhole_camera


class Xtion(pinhole_camera.PinholeCameraModel):

    """Camera model class for Xtion

    https://rosindustrial.org/news/2016/1/13/3d-camera-survey

    """

    def __init__(self):
        height = 480
        width = 640
        fovy = 45.0
        fovx = pinhole_camera.PinholeCameraModel.calc_fovx(fovy, height, width)
        fx = self.calc_f_from_fov(fovx, width)
        fy = self.calc_f_from_fov(fovy, height)
        K = [fx, 0, width / 2.0,
             0, fy, height / 2.0,
             0, 0, 1]
        P = [fx, 0, width / 2.0, 0,
             0, fy, height / 2.0, 0,
             0, 0, 1, 0]
        super(Xtion, self).__init__(
            image_height=height,
            image_width=width,
            K=K,
            P=P,
            name='xtion')
