#!/usr/bin/env python

import os
import sys
import platform
import glob

from distutils.core import setup, Extension
from torch.utils.ffi import create_extension

include_kenlm=True

third_party_libs = ["eigen3", "utf8"]
compile_args = ['-std=c++11', '-fPIC', '-w', '-O3', '-DNDEBUG']
if include_kenlm:
    third_party_libs.append("kenlm")
    compile_args.extend(['-DINCLUDE_KENLM', '-DKENLM_MAX_ORDER=6'])
    lib_sources = glob.glob('third_party/kenlm/util/*.cc') + glob.glob('third_party/kenlm/lm/*.cc') + glob.glob('third_party/kenlm/util/double-conversion/*.cc')
    lib_sources = [fn for fn in lib_sources if not (fn.endswith('main.cc') or fn.endswith('test.cc'))]
else:
    lib_sources = []
ext_libs = ['stdc++']

third_party_includes=["third_party/" + lib for lib in third_party_libs]
ctc_sources = ['pytorch_ctc/src/cpu_binding.cpp', 'pytorch_ctc/src/util/status.cpp']
ctc_headers = ['pytorch_ctc/src/cpu_binding.h',]

ffi = create_extension(
    name='ctc_decode',
    package=True,
    language='c++',
    headers=ctc_headers,
    sources=ctc_sources + lib_sources,
    include_dirs=third_party_includes,
    with_cuda=False,
    libraries=ext_libs,
    extra_compile_args=compile_args#, '-DINCLUDE_KENLM']
)
ffi = ffi.distutils_extension()
ffi.name = 'pytorch_ctc._ctc_decode'

setup(
    name="pytorch_ctc",
    version="0.1",
    description="CTC Decoder for PyTorch based on TensorFlow's implementation",
    url="https://github.com/ryanleary/pytorch-ctc-decode",
    author="Ryan Leary",
    author_email="ryanleary@gmail.com",
    # Require cffi.
    install_requires=["cffi>=1.0.0"],
    setup_requires=["cffi>=1.0.0"],
    # Exclude the build files.
    packages=["pytorch_ctc"],
    # Extensions to compile.
    ext_modules=[ffi]
)
