from cameramodels.pinhole_camera import PinholeCameraModel


class RGBDCamera(object):

    def __init__(self):
        self.rgb_camera = PinholeCameraModel()
        self.depth_camera = PinholeCameraModel()
