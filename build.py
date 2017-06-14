import os
import torch
from torch.utils.ffi import create_extension

this_file = os.path.dirname(__file__)

sources = ['ctc_decode/src/lib_ctc_decode.cc', 'ctc_decode/src/util/status.cc']
headers = []
defines = []

ffi = create_extension(
    name='_ext.ctc_decode',
    langauge='c++',
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=False,
    extra_compile_args=['-std=c++11', '-fPIC']
)

if __name__ == '__main__':
    ffi.build()
