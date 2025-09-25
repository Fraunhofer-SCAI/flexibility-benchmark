import os, subprocess
from setuptools import setup, find_packages


def read(fname):
    """Read a file to a string."""
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='flexbench',
    version=0.1,
    description='Benchmarks for system flexibility.',
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    keywords=[],
    # author=,
    # author_email=,
    # license=,
    url='https://gitlab.scai.fraunhofer.de/ndv/research/evolopro/flexbench',
    packages=find_packages(include=('flexbench')),
    # classifiers=[
    # "Development Status :: 4 - Beta",
    # "Programming Language :: Python :: 3",
    # "Intended Audience :: Science/Research",
    # "Intended Audience :: Developers",
    # "Topic :: Scientific/Engineering :: Artificial Intelligence",
    # ],
    install_requires=['pre-commit', 'numpy', 'deap', 'lmfit', 'pymoo', 'matplotlib', 'scikit-posthocs', 'pandas'],
    # tests_require=[
    # "pytest"
    # ],
)

subprocess.run('pre-commit install', shell=True)
