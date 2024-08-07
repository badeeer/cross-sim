from setuptools import setup, find_packages

setup(
    name='cross_sim_inference',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy==1.24.3',
        'scipy==1.11.1',
        'tensorflow==2.13.0',
        'ipython==8.8.0',
        'matplotlib==3.7.2',
        'cupy-cuda102==8.3.0',  # or 'cupy-cuda111==12.1.0' depending on your CUDA version
        'opencv-python==4.5.4',
        'torchvision==0.11.1',
        'larq==0.12.1',
    ],
    entry_points={
        'console_scripts': [
            # Define command-line scripts here if needed
        ],
    },
)

