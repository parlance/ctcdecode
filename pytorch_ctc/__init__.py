import torch

from cffi import FFI
ffi = FFI()
from ._ext.ctc_decode import ctc_beam_decode


def beam_decode(probs, seq_len=None, top_paths=1, beam_width=10, merge_repeated=True):
    prob_size = probs.size()
    max_seq_len = prob_size[0]
    batch_size = prob_size[1]
    num_classes = prob_size[2]

    if seq_len is not None and batch_size != seq_len.size():
        raise ValueError("seq_len shape must be a 1xbatch_size tensor or None")
    if top_paths < 1 or top_paths > beam_width:
        raise ValueError("top_paths must be greater than 1 and less than or equal to the beam_width")

    seq_len = torch.IntTensor(batch_size).zero_().add_(max_seq_len) if seq_len is None else seq_len
    output = torch.IntTensor(top_paths, batch_size, max_seq_len)
    scores = torch.FloatTensor(top_paths, batch_size)

    merge_int = 1 if merge_repeated else 0
    result = ctc_beam_decode(probs, seq_len, output, scores, top_paths, beam_width, merge_int)

    return output, scores
