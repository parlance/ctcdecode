#!/usr/bin/env python
import glob
import multiprocessing.pool
import os
import tarfile
import urllib.request
import warnings

from setuptools import setup, find_packages, distutils
from torch.utils.cpp_extension import BuildExtension
from torch.utils.cpp_extension import CppExtension, include_paths


def download_extract(url, dl_path):
    if not os.path.isfile(dl_path):
        # Already downloaded
        urllib.request.urlretrieve(url, dl_path)
    if dl_path.endswith(".tar.gz") and os.path.isdir(dl_path[:-len(".tar.gz")]):
        # Already extracted
        return
    tar = tarfile.open(dl_path)
    tar.extractall('third_party/')
    tar.close()


# Download/Extract openfst
download_extract('https://github.com/parlance/ctcdecode/releases/download/v1.0/openfst-1.6.7.tar.gz',
                 'third_party/openfst-1.6.7.tar.gz')


for file in ['third_party/ThreadPool/ThreadPool.h']:
    if not os.path.exists(file):
        warnings.warn('File `{}` does not appear to be present. Did you forget `git submodule update`?'.format(file))


compile_args = ['-O3', '-DKENLM_MAX_ORDER=6', '-std=c++14', '-fPIC']
ext_libs = []

third_party_libs = ["openfst-1.6.7/src/include", "ThreadPool"]

lib_sources = glob.glob('third_party/openfst-1.6.7/src/lib/*.cc')
lib_sources = [fn for fn in lib_sources if not (fn.endswith('main.cc') or fn.endswith('test.cc'))]

third_party_includes = [os.path.realpath(os.path.join("third_party", lib)) for lib in third_party_libs]
ctc_sources = glob.glob('ctcdecode/src/*.cpp')

extension = CppExtension(
    name='ctcdecode._ctc_decode',
    package=True,
    with_cuda=False,
    sources=ctc_sources + lib_sources,
    include_dirs=third_party_includes + include_paths(),
    libraries=ext_libs,
    extra_compile_args=compile_args,
    language='c++'
)


# monkey-patch for parallel compilation
# See: https://stackoverflow.com/a/13176803
def parallelCCompile(self,
                     sources,
                     output_dir=None,
                     macros=None,
                     include_dirs=None,
                     debug=0,
                     extra_preargs=None,
                     extra_postargs=None,
                     depends=None):
    # those lines are copied from distutils.ccompiler.CCompiler directly
    macros, objects, extra_postargs, pp_opts, build = self._setup_compile(
        output_dir, macros, include_dirs, sources, depends, extra_postargs)
    cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)

    # parallel code
    def _single_compile(obj):
        try:
            src, ext = build[obj]
        except KeyError:
            return
        self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)

    # convert to list, imap is evaluated on-demand
    thread_pool = multiprocessing.pool.ThreadPool(os.cpu_count())
    list(thread_pool.imap(_single_compile, objects))
    return objects


# hack compile to support parallel compiling
distutils.ccompiler.CCompiler.compile = parallelCCompile

setup(
    name="ctcdecode",
    version="1.0.2",
    description="CTC Decoder for PyTorch based on Paddle Paddle's implementation",
    url="https://github.com/parlance/ctcdecode",
    author="Ryan Leary",
    author_email="ryanleary@gmail.com",
    # Exclude the build files.
    packages=find_packages(exclude=["build"]),
    ext_modules=[extension],
    cmdclass={'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)}
)
