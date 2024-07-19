from setuptools import setup, find_packages

setup(
    name='gym_labyrinth',
    version='0.0.1',
    packages=find_packages(include=['gym_labyrinth', 'gym_labyrinth.*']),
    include_package_data=True,  # This line includes data files specified in MANIFEST.in
    install_requires=['gymnasium', 'numpy', 'opencv-python', 'pypot', 'sheeprl']
)
