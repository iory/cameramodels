# flake8: noqa

import pkg_resources


__version__ = pkg_resources.get_distribution('cameramodels').version


from cameramodels.pinhole_camera import PinholeCameraModel

import cameramodels.data
import cameramodels.models
