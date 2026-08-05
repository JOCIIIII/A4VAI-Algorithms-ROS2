from setuptools import setup

package_name = 'realgazebo_fastplanner_adapters'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config',
            ['config/goals_example.csv', 'config/goals_fullflight.csv']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kestrel',
    maintainer_email='kestrel@inha.edu',
    description='RealGazebo ↔ Fast-Planner input adapters (odom/cloud/goal).',
    license='MIT',
    entry_points={
        'console_scripts': [
            # 통합 시간정합 노드: odom+cloud 한 노드, cloud stamp 기반 pose 보간.
            'px4_lidar_to_odom_cloud = realgazebo_fastplanner_adapters.px4_lidar_to_odom_cloud:main',
            'csv_goal_pub = realgazebo_fastplanner_adapters.csv_goal_pub:main',
        ],
    },
)
