#!/usr/bin/env python

import os
import sys
import platform
import glob

from setuptools import setup, find_packages
from torch.utils.ffi import create_extension

this_file = os.path.dirname(__file__)

#Does gcc compile with this header and library?
def compile_test(header, library):
    dummy_path = os.path.join(os.path.dirname(__file__), "dummy")
    command = "bash -c \"g++ -include " + header + " -l" + library + " -x c++ - <<<'int main() {}' -o " + dummy_path + " >/dev/null 2>/dev/null && rm " + dummy_path + " 2>/dev/null\""
    return os.system(command) == 0

# this is a hack, but makes this significantly easier
include_kenlm = True
ex_klm = "--exclude-kenlm"
if ex_klm in sys.argv:
    include_kenlm=False
    sys.argv.remove(ex_klm)

third_party_libs = ["eigen3", "utf8"]
lib_sources = []
compile_args = ['-std=c++11', '-fPIC', '-w', '-O3', '-DNDEBUG', '-DTORCH_BINDING']
ext_libs = ['stdc++']

if compile_test('zlib.h', 'z'):
    compile_args.append('-DHAVE_ZLIB')
    ext_libs.append('z')

if compile_test('bzlib.h', 'bz2'):
    compile_args.append('-DHAVE_BZLIB')
    ext_libs.append('bz2')

if compile_test('lzma.h', 'lzma'):
    compile_args.append('-DHAVE_XZLIB')
    ext_libs.append('lzma')

if include_kenlm:
    third_party_libs.append("kenlm")
    compile_args.extend(['-DINCLUDE_KENLM', '-DKENLM_MAX_ORDER=6'])
    lib_sources = glob.glob('third_party/kenlm/util/*.cc') + glob.glob('third_party/kenlm/lm/*.cc') + glob.glob('third_party/kenlm/util/double-conversion/*.cc')
    lib_sources = [fn for fn in lib_sources if not (fn.endswith('main.cc') or fn.endswith('test.cc'))]

third_party_includes=[os.path.realpath(os.path.join("third_party", lib)) for lib in third_party_libs]
ctc_sources = ['ctcdecode/src/cpu_binding.cpp', 'ctcdecode/src/util/status.cpp']
ctc_headers = ['ctcdecode/src/cpu_binding.h',]

ffi = create_extension(
    name='ctcdecode._ext.ctc_decode',
    package=True,
    language='c++',
    headers=ctc_headers,
    sources=ctc_sources + lib_sources,
    include_dirs=third_party_includes,
    libraries=ext_libs,
    extra_compile_args=compile_args,
    relative_to=__file__,
    with_cuda=False
)
ffi = ffi.distutils_extension()

setup(
    name="ctcdecode",
    version="0.2",
    description="CTC Decoder for PyTorch based on TensorFlow's implementation",
    url="https://github.com/parlance/ctcdecode",
    author="Ryan Leary",
    author_email="ryanleary@gmail.com",
    # Require cffi.
    install_requires=["cffi>=1.0.0"],
    setup_requires=["cffi>=1.0.0"],
    # Exclude the build files.
    packages=find_packages(exclude=["build"]),
    # Extensions to compile.
    ext_modules=[ffi]

)
