import pkg_resources


__version__ = pkg_resources.get_distribution('cameramodels').version


from cameramodels.d415 import D415  # NOQA
from cameramodels.d435 import D435  # NOQA
from cameramodels.kinect_v2 import KinectV2  # NOQA
from cameramodels.pinhole_camera import PinholeCameraModel  # NOQA
from cameramodels.xtion import Xtion  # NOQA
