from __future__ import division

from cameramodels import pinhole_camera


class D435(pinhole_camera.PinholeCameraModel):

    """Camera model class for D435

    https://software.intel.com/en-us/realsense/d400

    """

    models = {'rgb':
              {'image_height': 1080,
               'image_width': 1920,
               'fovx': 69.4,
               'fovy': 42.5},
              'depth':
              {'image_height': 720,
               'image_width': 1280,
               'fovx': 91.2,
               'fovy': 65.5,
               'far': 10.0,
               'near': 0.11}}

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
        super(D435, self).__init__(
            image_height=height,
            image_width=width,
            K=K,
            P=P,
            name='d435')
