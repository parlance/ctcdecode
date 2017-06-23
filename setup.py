#!/usr/bin/env python

import os
import sys
import platform
import glob

from distutils.core import setup, Extension
from torch.utils.ffi import create_extension

third_party_includes=["third_party/" + lib for lib in ["eigen3", "utf8", "kenlm"]]

klm_files = glob.glob('third_party/kenlm/util/*.cc') + glob.glob('third_party/kenlm/lm/*.cc') + glob.glob('third_party/kenlm/util/double-conversion/*.cc')
klm_files = [fn for fn in klm_files if not (fn.endswith('main.cc') or fn.endswith('test.cc'))]

klm_args = ['-O3', '-DNDEBUG', '-DKENLM_MAX_ORDER=6', '-std=c++11']

print("klm_files:", klm_files)
print("klm_args:", klm_args)

klm_libs = ['stdc++']
if platform.system() != 'Darwin':
    klm_libs.append('rt')
    lib_ext = ".so"
else:
    lib_ext = ".dylib"

ctc_sources = ['pytorch_ctc/src/cpu_binding.cpp', 'pytorch_ctc/src/util/status.cpp']
ctc_headers = ['pytorch_ctc/src/cpu_binding.h']

ffi = create_extension(
    name='ctc_decode',
    package=True,
    language='c++',
    headers=ctc_headers,
    sources=ctc_sources + klm_files,
    include_dirs=third_party_includes,
    with_cuda=False,
    libraries=klm_libs,
    extra_compile_args=['-std=c++11', '-fPIC', '-w', '-DKENLM_MAX_ORDER=6', '-O3', '-DNDEBUG']
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
