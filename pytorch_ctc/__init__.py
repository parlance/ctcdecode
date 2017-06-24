import torch
import pytorch_ctc as ctc
from torch.utils.ffi import _wrap_function
from ._ctc_decode import lib as _lib, ffi as _ffi

__all__ = []


def _import_symbols(locals):
    for symbol in dir(_lib):
        fn = getattr(_lib, symbol)
        new_symbol = "_" + symbol
        locals[new_symbol] = _wrap_function(fn, _ffi)
        __all__.append(new_symbol)


_import_symbols(locals())


def beam_decode(probs, labels, seq_len=None, top_paths=1, beam_width=10, blank_index=0, space_index=28, merge_repeated=True, lm_path="", trie_path=""):
    prob_size = probs.size()
    max_seq_len = prob_size[0]
    batch_size = prob_size[1]
    num_classes = prob_size[2]

    if blank_index < 0 or blank_index >= num_classes:
        raise ValueError("blank_index must be within num_classes")
    if seq_len is not None and batch_size != seq_len.size(0):
        raise ValueError("seq_len shape must be a (batch_size) tensor or None")
    if top_paths < 1 or top_paths > beam_width:
        raise ValueError("top_paths must be greater than 1 and less than or equal to the beam_width")

    seq_len = torch.IntTensor(batch_size).zero_().add_(max_seq_len) if seq_len is None else seq_len
    output = torch.IntTensor(top_paths, batch_size, max_seq_len)
    scores = torch.FloatTensor(top_paths, batch_size)
    out_seq_len = torch.IntTensor(top_paths, batch_size)

    merge_int = 1 if merge_repeated else 0
    scorer = ctc._get_kenlm_scorer(labels, len(labels), space_index, blank_index, lm_path.encode(), trie_path.encode())
    decoder = ctc._get_ctc_beam_decoder(num_classes, batch_size, top_paths, beam_width, blank_index, merge_int, scorer, 1)
    print(scorer)
    result = ctc._ctc_beam_decode(probs, seq_len, output, scores, out_seq_len, decoder, 1)

    return output, scores, out_seq_len

def generate_lm_trie(dictionary_path, kenlm_path, output_path, labels, blank_index=0, space_index=28):
    result = _lib.generate_lm_trie(labels, len(labels), blank_index, space_index, kenlm_path.encode(), dictionary_path.encode(), output_path.encode())

    if result != 0:
        raise ValueError("Error encountered generating trie")
