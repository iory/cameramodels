from __future__ import print_function

import os
import shlex
import subprocess
import sys

from setuptools import find_packages
from setuptools import setup


version = '0.2.12'


if sys.argv[-1] == 'release':
    # Release via github-actions.
    commands = [
        'git tag v{:s}'.format(version),
        'git push origin master --tag',
    ]
    for cmd in commands:
        print('+ {}'.format(cmd))
        subprocess.check_call(shlex.split(cmd))
    sys.exit(0)


def listup_package_data():
    data_files = []
    for root, _, files in os.walk('cameramodels/data'):
        for filename in files:
            data_files.append(
                os.path.join(
                    root[len('cameramodels/'):],
                    filename))
    return data_files


setup_requires = []
install_requires = [
    'numpy',
    'pillow',
    'pyyaml',
    'scipy',
]

setup(
    name='cameramodels',
    version=version,
    description='A python library for cameramodels',
    author='iory',
    author_email='ab.ioryz@gmail.com',
    url='https://github.com/iory/cameramodels',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    license='MIT',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: Implementation :: CPython',
    ],
    packages=find_packages(),
    package_data={'cameramodels': listup_package_data()},
    zip_safe=False,
    setup_requires=setup_requires,
    install_requires=install_requires,
    entry_points={
        "console_scripts": [
            "resize-camera-info=cameramodels.apps.resize_camera_info:main"
        ]
    },
    extras_require={
        'all': [
            'open3d',
            'opencv-python;python_version>"2.7"',
            'opencv-python<=4.2.0.32;python_version<="2.7"',
        ],
    },
)
