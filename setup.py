#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from setuptools import find_packages, setup

# Package meta-static.
NAME = 'genz'
URL = 'https://github.com/ktavabi/genz'
AUTHOR = 'Kambiz Tavabi'
EMAIL = 'ktavabi@gmail.com'

REQUIRED = []

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the
# Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))
# Where the magic happens:
setup(
    name=NAME,
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    packages=find_packages(exclude=('tests',)),
    install_requires=REQUIRED,
    include_package_data=True,
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering'
        ],

    )
