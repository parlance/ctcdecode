import torch
import ctcdecode as ctc
from torch.utils.ffi import _wrap_function
from ._ext import ctc_decode
# from ._ext._ctc_decode import lib as _lib, ffi as _ffi
#
# __all__ = []
#
#
# def _import_symbols(locals):
#     for symbol in dir(_lib):
#         fn = getattr(_lib, symbol)
#         new_symbol = "_" + symbol
#         locals[new_symbol] = _wrap_function(fn, _ffi)
#         __all__.append(new_symbol)
#
#
# _import_symbols(locals())


class BaseCTCBeamDecoder(object):
    def __init__(self, labels, top_paths=1, beam_width=10, blank_index=0, space_index=28):
        self._labels = labels
        self._top_paths = top_paths
        self._beam_width = beam_width
        self._blank_index = blank_index
        self._space_index = space_index
        self._num_classes = len(labels)
        self._decoder = None

        if blank_index < 0 or blank_index >= self._num_classes:
            raise ValueError("blank_index must be within num_classes")

        if top_paths < 1 or top_paths > beam_width:
            raise ValueError("top_paths must be greater than 1 and less than or equal to the beam_width")

    def decode(self, probs, seq_len=None):
        prob_size = probs.size()
        max_seq_len = prob_size[0]
        batch_size = prob_size[1]
        num_classes = prob_size[2]

        if seq_len is not None and batch_size != seq_len.size(0):
            raise ValueError("seq_len shape must be a (batch_size) tensor or None")

        seq_len = torch.IntTensor(batch_size).zero_().add_(max_seq_len) if seq_len is None else seq_len
        output = torch.IntTensor(self._top_paths, batch_size, max_seq_len)
        scores = torch.FloatTensor(self._top_paths, batch_size)
        out_seq_len = torch.IntTensor(self._top_paths, batch_size)
        alignments = torch.IntTensor(self._top_paths, batch_size, max_seq_len)
        char_probs = torch.FloatTensor(self._top_paths, batch_size, max_seq_len)

        result = ctc_decode.ctc_beam_decode(self._decoder, self._decoder_type, probs, seq_len, output, scores, out_seq_len,
                                      alignments, char_probs)

        return output, scores, out_seq_len, alignments, char_probs


class BaseScorer(object):
    def __init__(self):
        self._scorer_type = 0
        self._scorer = None

    def get_scorer_type(self):
        return self._scorer_type

    def get_scorer(self):
        return self._scorer


class Scorer(BaseScorer):
    def __init__(self):
        super(Scorer, self).__init__()
        self._scorer = ctc_decode.get_base_scorer()


class DictScorer(BaseScorer):
    def __init__(self, labels, trie_path, blank_index=0, space_index=28):
        super(DictScorer, self).__init__()
        self._scorer_type = 1
        self._scorer = ctc_decode.get_dict_scorer(labels, len(labels), space_index, blank_index, trie_path.encode())

    def set_min_unigram_weight(self, weight):
        if weight is not None:
            ctc_decode.set_dict_min_unigram_weight(self._scorer, weight)


class KenLMScorer(BaseScorer):
    def __init__(self, labels, lm_path, trie_path, blank_index=0, space_index=28):
        super(KenLMScorer, self).__init__()
        if ctc_decode.kenlm_enabled() != 1:
            raise ImportError("ctcdecode not compiled with KenLM support.")
        self._scorer_type = 2
        self._scorer = ctc_decode.get_kenlm_scorer(labels, len(labels), space_index, blank_index, lm_path.encode(),
                                             trie_path.encode())

    # This is a way to make sure the destructor is called for the C++ object
    # Frees all the member data items that have allocated memory
    def __del__(self):
        ctc_decode.free_kenlm_scorer(self._scorer)

    def set_lm_weight(self, weight):
        if weight is not None:
            ctc_decode.set_kenlm_scorer_lm_weight(self._scorer, weight)

    def set_word_weight(self, weight):
        if weight is not None:
            ctc_decode.set_kenlm_scorer_wc_weight(self._scorer, weight)

    def set_min_unigram_weight(self, weight):
        if weight is not None:
            ctc_decode.set_kenlm_min_unigram_weight(self._scorer, weight)


class CTCBeamDecoder(BaseCTCBeamDecoder):
    def __init__(self, scorer, labels, top_paths=1, beam_width=10, blank_index=0, space_index=28):
        super(CTCBeamDecoder, self).__init__(labels, top_paths=top_paths, beam_width=beam_width,
                                             blank_index=blank_index, space_index=space_index)
        self._scorer = scorer
        self._decoder_type = self._scorer.get_scorer_type()
        self._decoder = ctc_decode.get_ctc_beam_decoder(self._num_classes, top_paths, beam_width, blank_index,
                                                        self._scorer.get_scorer(), self._decoder_type)

    def set_label_selection_parameters(self, label_size=0, label_margin=-1):
        ctc_decode.set_label_selection_parameters(self._decoder, label_size, label_margin)


def generate_lm_dict(dictionary_path, output_path, labels, kenlm_path=None, blank_index=0, space_index=28):
    if kenlm_path is not None and ctc_decode.kenlm_enabled() != 1:
        raise ImportError("ctcdecode not compiled with KenLM support.")
    result = None
    if kenlm_path is not None:
        result = ctc_decode.generate_lm_dict(labels, len(labels), blank_index, space_index, kenlm_path.encode(),
                                             dictionary_path.encode(), output_path.encode())
    else:
        result = ctc_decode.generate_dict(labels, len(labels), blank_index, space_index,
                                          dictionary_path.encode(), output_path.encode())
    if result != 0:
        raise ValueError("Error encountered generating dictionary")
