from setuptools import setup
from setuptools import find_packages

setup(
    name="superglue",
    version="0.1",
    description="Pre-trained model and PyTorch implementation of SuperGlue: Learning Feature Matching with Graph Neural Networks.",
    author="Nurenda Technologies",
    author_email="info@nurenda.com",
    packages=find_packages(include=["superglue", "superglue.*"]),
)
