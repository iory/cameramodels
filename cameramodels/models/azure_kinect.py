from __future__ import division

from enum import IntEnum

from cameramodels import pinhole_camera


class K4AColorResolution(IntEnum):
    K4A_COLOR_RESOLUTION_OFF = 0
    K4A_COLOR_RESOLUTION_720P = 1  # 1280 * 720  16:9
    K4A_COLOR_RESOLUTION_1080P = 2  # 1920 * 1080 16:9
    K4A_COLOR_RESOLUTION_1440P = 3  # 2560 * 1440 16:9
    K4A_COLOR_RESOLUTION_1536P = 4  # 2048 * 1536 4:3
    K4A_COLOR_RESOLUTION_2160P = 5  # 3840 * 2160 16:9
    K4A_COLOR_RESOLUTION_3072P = 6  # 4096 * 3072 4:3


color_resolutions = ('720P', '1080P', '1440P', '1536P', '2160P', '3072P',)
depth_modes = ('NFOV_2X2BINNED', 'NFOV_UNBINNED',
               'WFOV_2X2BINNED', 'WFOV_UNBINNED',
               # 'PASSIVE_IR',
               )


class AzureKinect(pinhole_camera.PinholeCameraModel):

    """Camera model class for Azure Kinect

    https://docs.microsoft.com/en-us/azure/Kinect-dk/hardware-specification

    """

    models = {'rgb':
              {'720P':
               {'image_height': 720,
                'image_width': 1280,
                'fovx': 90,
                'fovy': 59},
               '1080P':
               {'image_height': 1080,
                'image_width': 1920,
                'fovx': 90,
                'fovy': 59},
               '1440P':
               {'image_height': 1440,
                'image_width': 2560,
                'fovx': 90,
                'fovy': 59},
               '1536P':
               {'image_height': 1536,
                'image_width': 2048,
                'fovx': 90,
                'fovy': 74.3},
               '2160P':
               {'image_height': 2160,
                'image_width': 3840,
                'fovx': 90,
                'fovy': 59},
               '3072P':
               {'image_height': 3072,
                'image_width': 4096,
                'fovx': 90,
                'fovy': 74.3}},
              'depth':
              {'NFOV_2X2BINNED':
               {'image_height': 288,
                'image_width': 320,
                'fovx': 75,
                'fovy': 65,
                'far': 5.46,
                'near': 0.5},
               'NFOV_UNBINNED':
               {'image_height': 576,
                'image_width': 640,
                'fovx': 75,
                'fovy': 65,
                'far': 3.86,
                'near': 0.5},
               'WFOV_2X2BINNED':
               {'image_height': 512,
                'image_width': 512,
                'fovx': 120,
                'fovy': 120,
                'far': 2.88,
                'near': 0.25},
               'WFOV_UNBINNED':
               {'image_height': 1024,
                'image_width': 1024,
                'fovx': 120,
                'fovy': 120,
                'far': 2.21,
                'near': 0.25}}}

    def __init__(self, mode='rgb', resolution='720P',
                 depth_mode='NFOV_UNBINNED'):
        if mode not in self.models:
            raise ValueError
        if mode == 'rgb':
            if resolution not in color_resolutions:
                raise ValueError('Valid color resollution are as follows {}'.
                                 format(color_resolutions))
            model = self.models[mode][resolution]
        elif mode == 'depth':
            if depth_mode not in depth_modes:
                raise ValueError('Valid depth modes are as follows {}'.
                                 format(depth_modes))
            model = self.models[mode][depth_mode]

        height = model['image_height']
        width = model['image_width']
        fovy = model['fovy']
        fovx = model['fovx']

        fx = self.calc_f_from_fov(fovx, width)
        fy = self.calc_f_from_fov(fovy, height)
        K = [fx, 0, width / 2.0,
             0, fy, height / 2.0,
             0, 0, 1]
        P = [fx, 0, width / 2.0, 0,
             0, fy, height / 2.0, 0,
             0, 0, 1, 0]
        super(AzureKinect, self).__init__(
            image_height=height,
            image_width=width,
            K=K,
            P=P,
            name='azure_kinect')
