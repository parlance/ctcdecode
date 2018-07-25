#!/usr/bin/env python

import glob
import os
import tarfile
import warnings

import wget
from torch.utils.ffi import create_extension


def download_extract(url, dl_path):
    if not os.path.isfile(dl_path):
        # Already downloaded
        wget.download(url, out=dl_path)
    if dl_path.endswith(".tar.gz") and os.path.isdir(dl_path[:-len(".tar.gz")]):
        # Already extracted
        return
    tar = tarfile.open(dl_path)
    tar.extractall('third_party/')
    tar.close()


# Download/Extract openfst, boost
download_extract('https://sites.google.com/site/openfst/home/openfst-down/openfst-1.6.7.tar.gz',
                 'third_party/openfst-1.6.7.tar.gz')
download_extract('https://dl.bintray.com/boostorg/release/1.67.0/source/boost_1_67_0.tar.gz',
                 'third_party/boost_1_67_0.tar.gz')

for file in ['third_party/kenlm/setup.py', 'third_party/ThreadPool/ThreadPool.h']:
    if not os.path.exists(file):
        warnings.warn('File `{}` does not appear to be present. Did you forget `git submodule update`?'.format(file))


# Does gcc compile with this header and library?
def compile_test(header, library):
    dummy_path = os.path.join(os.path.dirname(__file__), "dummy")
    command = "bash -c \"g++ -include " + header + " -l" + library + " -x c++ - <<<'int main() {}' -o " + dummy_path \
              + " >/dev/null 2>/dev/null && rm " + dummy_path + " 2>/dev/null\""
    return os.system(command) == 0


compile_args = ['-O3', '-DNDEBUG', '-DKENLM_MAX_ORDER=6', '-std=c++11', '-fPIC', '-std=c99', '-w']
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

third_party_libs = ["kenlm", "openfst-1.6.7/src/include", "ThreadPool", "boost_1_67_0", "utf8"]
compile_args.extend(['-DINCLUDE_KENLM', '-DKENLM_MAX_ORDER=6'])
lib_sources = glob.glob('third_party/kenlm/util/*.cc') + glob.glob('third_party/kenlm/lm/*.cc') + glob.glob(
    'third_party/kenlm/util/double-conversion/*.cc') + glob.glob('third_party/openfst-1.6.7/src/lib/*.cc')
lib_sources = [fn for fn in lib_sources if not (fn.endswith('main.cc') or fn.endswith('test.cc'))]

third_party_includes = [os.path.realpath(os.path.join("third_party", lib)) for lib in third_party_libs]
ctc_sources = glob.glob('ctcdecode/src/*.cpp')
ctc_headers = ['ctcdecode/src/binding.h', ]

ffi = create_extension(
    name='ctcdecode._ext.ctc_decode',
    package=True,
    language='c++',
    headers=ctc_headers,
    sources=ctc_sources + lib_sources,
    include_dirs=third_party_includes,
    with_cuda=False,
    libraries=ext_libs,
    extra_compile_args=compile_args
)

if __name__ == '__main__':
    ffi.build()
