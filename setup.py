#!/usr/bin/env python
try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup

setup(
    name="catapalt",
    version="0.0",
    description="Python tools for catalyst dataset analysis",
    author="Brook Wander",
    author_email="bwander@andrew.cmu.edu",
    url="https://github.com/ulissigroup/catapalt",
    packages=find_packages(),
)
