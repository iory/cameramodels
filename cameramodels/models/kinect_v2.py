from __future__ import division

from cameramodels import pinhole_camera


class KinectV2(pinhole_camera.PinholeCameraModel):

    """Camera model class for Kinect v2

    """

    models = {'rgb':
              {'image_height': 1080,
               'image_width': 1920,
               'fovx': 84.1,
               'fovy': 53.8},
              'depth':
              {'image_height': 424,
               'image_width': 512,
               'fovx': 70.6,
               'fovy': 60,
               'far': 4.5,
               'near': 0.5}}

    def __init__(self, mode='rgb'):
        if mode not in self.models:
            raise ValueError
        height = self.models[mode]['image_height']
        width = self.models[mode]['image_width']
        fovy = self.models[mode]['fovy']
        fovx = self.models[mode]['fovx']

        fx = self.calc_f_from_fov(fovx, width)
        fy = self.calc_f_from_fov(fovy, height)
        K = [fx, 0, width / 2.0,
             0, fy, height / 2.0,
             0, 0, 1]
        P = [fx, 0, width / 2.0, 0,
             0, fy, height / 2.0, 0,
             0, 0, 1, 0]
        super(KinectV2, self).__init__(
            image_height=height,
            image_width=width,
            K=K,
            P=P,
            name='kinect_v2')
