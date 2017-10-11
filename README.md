# pytorch-ctc
PyTorch-CTC is an implementation of CTC (Connectionist Temporal Classification) beam search decoding for PyTorch. C++ code borrowed liberally from TensorFlow with some improvements to increase flexibility.

## Installation
The library is largely self-contained and requires only PyTorch and CFFI. Building the C++ library requires gcc. KenLM language modeling support is also optionally included, and enabled by default.

```bash
# get the code
git clone --recursive https://github.com/ryanleary/pytorch-ctc.git
cd pytorch-ctc

# install dependencies (PyTorch and CFFI)
pip install -r requirements.txt

python setup.py install
# If you do NOT require kenlm, the `--recursive` flag is not required on git clone
# and `--exclude-kenlm` should be appended to the `python setup.py install` command
```

## API
pytorch-ctc includes a CTC beam search decoder with multiple scorer implementations. A `scorer` is a function that the decoder calls to condition the probability of a given beam based on its state.

### Scorers
Two Scorer implementations are currently implemented for pytorch-ctc.

**Scorer:** is a NO-OP and enables the decoder to do a vanilla beam decode
```python
scorer = Scorer()
```

**KenLMScorer:** conditions beams based on the provided KenLM binary language model.
```python
scorer = KenLMScorer(labels, lm_path, trie_path, blank_index=0, space_index=28)
```

where:
- `labels` is a string of output labels given in the same order as the output layer
- `lm_path` path to a binary KenLM language model for decoding
- `trie_path` path to a Trie containing the lexicon (see generate_lm_trie)
- `blank_index` is used to specify which position in the output distribution represents the `blank` class
- `space_index` is used to specify which position in the output distribution represents the word separator class

The `KenLMScorer` may be further configured with weights for the language model contribution to the score (`lm_weight`), as well as word and valid word bonuses (to offset decreasing probability as a function of sequence length).

```python
scorer.set_lm_weight(2.1)
scorer.set_word_weight(1.1)
scorer.set_valid_word_weight(1.5)
```

### Decoder
```python
decoder = CTCBeamDecoder(scorer, labels, top_paths=3, beam_width=20,
                         blank_index=0, space_index=28, merge_repeated=False)
```

where:
- `scorer` is an instance of a concrete implementation of the `BaseScorer` class
- `labels` is a string of output labels given in the same order as the output layer
- `top_paths` is used to specify how many hypotheses to return
- `beam_width` is the number of beams to evaluate in a given step
- `blank_index` is used to specify which position in the output distribution represents the `blank` class
- `space_index` is used to specify which position in the output distribution represents the word separator class
- `merge_repeated` if True will collapse repeated characters

```python
output, score, out_seq_len = decoder.decode(probs, sizes=None)
```

where:
- `probs` is a FloatTensor of log-probabilities with shape `(seq_len, batch_size, num_classes)`
- `seq_len` is an optional IntTensor of integer sequence lengths with shape `(batch_size)`

and returns:
- `output` is an IntTensor of character classes of shape `(top_paths, batch_size, seq_len)`
- `score` is a FloatTensor of log-probabilities representing the likelihood of the transcription with shape `(top_paths, batch_size)`
- `out_seq_len` is an IntTensor containing the length of the output sequence with shape `(top_paths, batch_size)`

### Utilities
```python
generate_lm_trie(dictionary_path, kenlm_path, output_path, labels, blank_index, space_index)
```

A vocabulary trie is required for the KenLM Scorer. The trie is created from a lexicon specified as a newline separated text file of words in the vocabulary.

## Acknowledgements
Thanks to [ebrevdo](https://github.com/ebrevdo) for the original TensorFlow CTC decoder implementation, [timediv](https://github.com/timediv) for his KenLM extension, and [SeanNaren](https://github.com/seannaren) for his assistance.
