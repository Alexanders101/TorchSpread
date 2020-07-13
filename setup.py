from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='torch_spread',
    version='1.1',
    packages=['torch_spread'],
    url='https://github.com/Alexanders101/TorchSpread',
    license='MIT',
    author='Alexander Shmakov',
    author_email='Alexanders101@gmail.com',
    description='Library for distributed reinforcement learning and dynamic batching with pytorch',
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)