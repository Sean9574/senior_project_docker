from glob import glob

from setuptools import find_packages, setup

package_name = 'senior_project'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(include=[package_name, f"{package_name}.*"]),
    data_files=[
        ('share/ament_index/resource_index/packages', [f'resource/{package_name}']),
        (f'share/{package_name}', ['package.xml']),
        # Only install *.launch.py once (no duplicate glob for *.py)
        (f'share/{package_name}/launch', glob('launch/*.launch.py')),
        (f'share/{package_name}/modules', glob('modules/*.py')),
        (f'share/{package_name}/scripts', glob('scripts/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sean',
    maintainer_email='sean9574@gmail.com',
    description='Stretch RL training + sim bridge',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'learner_node = senior_project.learner_node:main',
            'sam3_ros_node = senior_project.sam3_ros_node:main',
            'sam3_room_scanner = senior_project.sam3_room_scanner:main',
            'sam3_goal_generator = senior_project.sam3_goal_generator:main',
            'sam3_depth_node = senior_project.sam3_depth_node:main',
            'sam3_server = senior_project.sam3_server:main',
            
            
        ],
    },
)
