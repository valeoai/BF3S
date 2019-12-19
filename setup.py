from setuptools import setup
from setuptools import find_packages

setup(
    name="BF3S",
    version="0.0.1",
    description="Boosting Few-Shot Visual Learning with Self-Supervision",
    author="Spyros Gidaris",
    packages=find_packages(),
    install_requires=["tensorboardX",
                      "tqdm",
                      "numpy",
                      "torch",
                      "torchvision",
                      "Pillow",
                      'torchnet'],
)
