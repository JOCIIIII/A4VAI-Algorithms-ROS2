from setuptools import find_packages, setup

package_name = 'pathfollowing'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kestrel',
    maintainer_email='kestrel@inha.edu',
    description='TODO: Package description',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'node_MPPI_output = pathfollowing.node_MPPI_output:main',
            'node_att_ctrl = pathfollowing.node_att_ctrl:main',
        ],
    },
)
