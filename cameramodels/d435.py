from cameramodels import pinhole_camera


class D435(pinhole_camera.PinholeCameraModel):

    """Camera model class for D435

    https://software.intel.com/en-us/realsense/d400

    """

    models = {'rgb':
              {'image_height': 1920,
               'image_width': 1080,
               'fov': 69.4},
              'depth':
              {'image_height': 1280,
               'image_width': 720,
               'fov': 87.0,
               'far': 10.0,
               'near': 0.11}}

    def __init__(self, mode='rgb'):
        if mode not in self.models:
            raise ValueError
        height = self.models[mode]['image_height']
        width = self.models[mode]['image_width']
        fovy = self.models[mode]['fov']
        aspect = width / height

        fovx = aspect * fovy
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
            P=P)
