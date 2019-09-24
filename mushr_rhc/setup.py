#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="MuSHR RHC",
    version="1.0",
    description="MuSHR receding horizon control library.",
    author="Matthew Rockett & Johan Michalove",
    author_email="mushr@cs.washington.edu",
    url="http://mushr.io",
    packages=find_packages(),
    install_requires=["numpy", "torch"],
    license="BSD-3",
)
