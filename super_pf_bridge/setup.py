from setuptools import setup

package_name = 'super_pf_bridge'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yonghajo',
    maintainer_email='jociiiii@inha.edu',
    description='SUPER MINCO polynomial trajectory (ENU) -> A4VAI PathFollowing '
                'LocalWaypointSetpoint (NED) bridge. Mirrors fastplanner_pf_bridge '
                'output side; input is mars_quadrotor_msgs/PolynomialTrajectory.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'poly_to_waypoints = super_pf_bridge.poly_to_waypoints:main',
        ],
    },
)
