# use "pip install -e ." to run setup

from setuptools import setup, find_packages

setup(name='Evolving-an-SNN',
      version='0.1',
      packages=find_packages(),
      install_requires = ["numpy>=1.16.4",
                          "tqdm>=4.31.1",
                          "Pillow>=5.4.1",
                          "matplotlib>=3.0.3",
                          "networkx>=2.2c",
                          "gudhi>=2.3.0"])
