from setuptools import find_packages, setup

package_name = 'trossen_sim2real_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hayato',
    maintainer_email='hayato.kojima80@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'sim2real_node = trossen_sim2real_ros.sim2real_node:main',
            'sim2real_node_trapezoidal = trossen_sim2real_ros.sim2real_node_trapezoidal:main',
            'sim2real_node_switch = trossen_sim2real_ros.sim2real_node_switch:main',
            'sim2real_node_test = trossen_sim2real_ros.sim2real_node_test:main',
        ],
    },
)
