#!/usr/bin/env python3

import os
from setuptools import setup

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
    packages=['tfx_gpt2'],
    include_package_data=True,
    python_requires=">=3.6.*",
    install_requires=["fire>=0.1.3",
                      "regex==2017.4.5"
                      "requests==2.21.0",
                      "tqdm==4.31.1",
                      "toposort==1.5",
                      "tensorflow-gpu==1.12.0"],
    license=about['__license__'],
    zip_safe=False,
    entry_points={
        'console_scripts': ['py-package-template=tfx_gpt2.entry_points:main'],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3.7',
    ],
    keywords='package development template'
)
