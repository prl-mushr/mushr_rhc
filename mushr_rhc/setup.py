#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="mushr-rhc",
    version="1.0",
    description="MuSHR receding horizon control library.",
    author="Matthew Rockett & Johan Michalove",
    author_email="mushr@cs.washington.edu",
    url="http://mushr.io",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "networkx",
        "scipy",
        "sklearn",
    ],
    license="BSD-3",
)
