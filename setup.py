#!/usr/bin/env python3

import os
import setuptools
from setuptools import setup, find_packages

# get key package details from tfx_gpt2/__version__.py
about = {}  # type: ignore
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'tfx_gpt2', '__version__.py')) as f:
    exec(f.read(), about)

# load the README file and use it as the long_description for PyPI
with open('README.md', 'r') as f:
    readme = f.read()

# package configuration - for reference see:
# https://setuptools.readthedocs.io/en/latest/setuptools.html#id9
setup(
    name=about['__title__'],
    description=about['__description__'],
    long_description=readme,
    long_description_content_type='text/markdown',
    version=about['__version__'],
    author=about['__author__'],
    author_email=about['__author_email__'],
    url=about['__url__'],
    packages=setuptools.find_packages(),
    include_package_data=True,
    python_requires=">=3.6.*",
    install_requires=["regex",
                      "requests",
                      "tqdm",
                      "toposort==1.5",
                      "tensorflow==1.15.0",
                      "tensorflow-serving-api==1.15.0"
                      "tfx==0.15.0",
                      "pyarrow==0.14.1",
                      "apache-beam==2.16.0",
                      "mlflow",
                      "pymongo"],
    license=about['__license__'],
    zip_safe=False,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3.7',
    ],
    keywords='package development template'
)
