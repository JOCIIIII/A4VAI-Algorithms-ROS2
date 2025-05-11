from setuptools import setup

package_name = 'offboard_ctrl'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='wetech@kaist.ac.kr',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "node_att_ctrl = offboard_ctrl.node_att_ctrl:main",
            "node_MPPI_output = offboard_ctrl.node_MPPI_output:main",
            "node_GPR_output = offboard_ctrl.node_MPPI_output:main",
            "node_pos_ctrl = offboard_ctrl.node_pos_ctrl:main",
            "node_accel_ctrl = offboard_ctrl.node_accel_ctrl:main",
            "node_att_ctrl_w_accel = offboard_ctrl.node_att_ctrl_w_accel:main"
        ],
    },
)
