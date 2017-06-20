# pytorch-ctc
Implementation of CTC (Connectionist Temporal Classification) beam search decoding with PyTorch bindings. C++ code borrowed liberally from TensorFlow with some improvements to increase flexibility.

## Installation
The library is largely self-contained and requires only PyTorch and CFFI. Building the C++ library requires at least GCC-5. If gcc-5 or later is not your default compiler, you may specify the path via environment variables.

```bash
# get the code
git clone https://github.com/ryanleary/pytorch-ctc.git
cd pytorch-ctc

# install dependencies (PyTorch and CFFI)
pip install -r requirements.txt

# build the extension and install python package
# python setup.py install
CC=/path/to/gcc-5 CXX=/path/to/g++-5 python setup.py install
```

## API
pytorch-ctc currently supports a single method:
```python
output, score, out_seq_len = beam_decode(probs, seq_len=None, top_paths=1, beam_width=10,
                                         blank_index=0, merge_repeated=True)
```

where:
- `probs` is a FloatTensor of log-probabilities with shape `(seq_len, batch_size, num_classes)`
- `seq_len` is an optional IntTensor of integer sequence lengths with shape `(batch_size)`
- `top_paths` is used to specify how many hypotheses to return
- `beam_width` is the number of beams to evaluate in a given step
- `blank_index` is used to specify which position in the output distribution represents the `blank` class
- `merge_repeated` if True will collapse repeated characters

and returns:
- `output` is an IntTensor of character classes of shape `(top_paths, batch_size, seq_len)`
- `score` is a FloatTensor of log-probabilities representing the likelihood of the transcription with shape `(top_paths, batch_size)`
- `out_seq_len` is an IntTensor containing the length of the output sequence with shape `(top_paths, batch_size)`
