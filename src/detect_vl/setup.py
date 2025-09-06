from setuptools import find_packages, setup
import os
from glob import glob
package_name = 'detect_vl'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/scripts', glob('scripts/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='epon',
    maintainer_email='2698678406@qq.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'run_detect_node = detect_vl.run_detect:main',
            'start_service = detect_vl.start_service:main',
            'ros_visualize_map = detect_vl.ros_visualize_map:main',
            'yolo11_object365_detector = detect_vl.yolo11_365_object_detection:main',
            'test_yolo11_integration = detect_vl.scripts.test_yolo11_integration:main',
        ],
    },
)
