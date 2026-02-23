from setuptools import setup
import os
from glob import glob

package_name = 'ur5_minimal_moveit'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    entry_points={
        'console_scripts': [
            'fake_joint_trajectory_controller = ur5_minimal_moveit.fake_joint_trajectory_controller:main',
        ],
    },
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='Minimal MoveIt launch for UR5',
    license='MIT',
    tests_require=['pytest'],
)
