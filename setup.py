from __future__ import print_function

import distutils.spawn
import shlex
import subprocess
import sys

from setuptools import find_packages
from setuptools import setup


version = '0.1.3'


if sys.argv[-1] == 'release':
    if not distutils.spawn.find_executable('twine'):
        print(
            'Please install twine:\n\n\tpip install twine\n',
            file=sys.stderr,
        )
        sys.exit(1)

    commands = [
        'git tag v{:s}'.format(version),
        'git push origin master --tag',
        'python setup.py sdist',
        'twine upload dist/cameramodels-{:s}.tar.gz'.format(version),
    ]
    for cmd in commands:
        print('+ {}'.format(cmd))
        subprocess.check_call(shlex.split(cmd))
    sys.exit(0)


setup_requires = []
install_requires = [
    'numpy',
    'pyyaml',
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
    zip_safe=False,
    setup_requires=setup_requires,
    install_requires=install_requires,
    extras_require={
        'all': ['open3d'],
    },
)
