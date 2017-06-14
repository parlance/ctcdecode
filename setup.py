#!/usr/bin/env python

import os
import sys

from setuptools import setup, find_packages
from torch.utils.ffi import create_extension

this_file = os.path.dirname(__file__)

sources = ['ctc_decode/src/cpu_binding.cpp', 'ctc_decode/src/util/status.cpp']
headers = ['ctc_decode/src/cpu_binding.h']
defines = []

ffi = create_extension(
    name='ctc_decode',
    language='c++',
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=False,
    extra_compile_args=['-std=c++11', '-fPIC']
)
ffi = ffi.distutils_extension()
ffi.name = 'pytorch_ctc_decode._ctc_decode'

setup(
    name="pytorch_ctc_decode",
    version="0.1",
    description="CTC Decoder for PyTorch based on TensorFlow's implementation",
    url="https://github.com/ryanleary/pytorch-ctc-decode",
    author="Ryan Leary",
    author_email="ryanleary@gmail.com",
    # Require cffi.
    install_requires=["cffi>=1.0.0"],
    setup_requires=["cffi>=1.0.0"],
    # Exclude the build files.
    packages=find_packages(exclude=["build"]),
    # Package where to put the extensions. Has to be a prefix of build.py.
    ext_package="",
    # Extensions to compile.
    ext_modules=[ffi]
)
