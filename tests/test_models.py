import unittest

from cameramodels.models import azure_kinect
from cameramodels.models import AzureKinect
from cameramodels.models import D415
from cameramodels.models import D435
from cameramodels.models import KinectV2
from cameramodels.models import Xtion


class TestModels(unittest.TestCase):

    def test_azure_kinect(self):
        for res in azure_kinect.color_resolutions:
            AzureKinect(mode='rgb', resolution=res)

        for dm in azure_kinect.depth_modes:
            AzureKinect(mode='depth', depth_mode=dm)

    def test_d415(self):
        D415(mode='rgb')
        D415(mode='depth')

    def test_d435(self):
        D435(mode='rgb')
        D435(mode='depth')

    def test_kinectv2(self):
        KinectV2(mode='rgb')
        KinectV2(mode='depth')

    def test_xtion(self):
        Xtion()
