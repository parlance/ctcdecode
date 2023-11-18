from time import time
from typing import List, Optional, Union

import torch

from ._ext import ctc_decode


class CTCBeamDecoder(object):
    """
    PyTorch wrapper for DeepSpeech PaddlePaddle Beam Search Decoder.
    Args:
        labels (list): The tokens/vocab used to train your model.
                        They should be in the same order as they are in your model's outputs.
        model_path (basestring): The path to your external KenLM language model(LM)
        alpha (float): Weighting associated with the LMs probabilities.
                        A weight of 0 means the LM has no effect.
        beta (float):  Weight associated with the number of words within our beam.
        cutoff_top_n (int): Cutoff number in pruning. Only the top cutoff_top_n characters
                            with the highest probability in the vocab will be used in beam search.
        cutoff_prob (float): Cutoff probability in pruning. 1.0 means no pruning.
        beam_width (int): This controls how broad the beam search is. Higher values are more likely to find top beams,
                            but they also will make your beam search exponentially slower.
        num_processes (int): Parallelize the batch using num_processes workers.
        blank_id (int): Index of the CTC blank token (probably 0) used when training your model.
        log_probs_input (bool): False if your model has passed through a softmax and output probabilities sum to 1.
        is_bpe_based (bool): True if your labels contains bpe tokens else False
        lm_type (str): Whether the language model file is character, bpe or word based
        token_separator (str): prefix of the bpe tokens. Default value is "#" and it is always assumed that the tokens
            starting with this prefix are meant to be merged with tokens that doesn't contain this prefix
    """

    def __init__(
        self,
        labels: List[str],
        model_path: Optional[str] = None,
        alpha: float = 0,
        beta: float = 0,
        cutoff_top_n: int = 40,
        cutoff_prob: float = 1.0,
        beam_width: int = 100,
        num_processes: int = 4,
        blank_id: int = 0,
        log_probs_input: bool = False,
        is_bpe_based: bool = False,
        unk_score: float = -5.0,
        lm_type: str = "character",
        token_separator: str = "#",
        lexicon_fst_path: Optional[str] = None,
    ):
        self.cutoff_top_n = cutoff_top_n
        self._beam_width = beam_width
        self._scorer = None
        self._num_processes = num_processes
        self._labels = list(labels)  # Ensure labels are a list
        self._num_labels = len(labels)
        self._blank_id = blank_id
        self._log_probs = True if log_probs_input else False
        self.token_separator = token_separator

        lexicon_fst_path = lexicon_fst_path if lexicon_fst_path is not None else ""

        if model_path:
            self._scorer = ctc_decode.paddle_get_scorer(
                alpha,
                beta,
                model_path.encode(),
                self._labels,
                lm_type,
                lexicon_fst_path.encode(),
            )
        self._is_bpe_based = is_bpe_based
        self._cutoff_prob = cutoff_prob

        self.decoder_options = ctc_decode.paddle_get_decoder_options(
            self._labels,
            cutoff_top_n,
            cutoff_prob,
            beam_width,
            num_processes,
            blank_id,
            self._log_probs,
            is_bpe_based,
            unk_score,
            token_separator,
        )

    def create_hotword_scorer(
        self,
        hotwords: List[List[str]],
        hotword_weight: Union[float, List[float]] = 10.0,
    ):
        """
        Method to create hotword scorer object for the given hotwords
        Args:
        hotwords (List[List[str]]) - Tokenized list of hotwords.
            For example:
                Hotword list for BPE token inputs = [ ["co", "##r", "##p"], ["t", "##es","##t"] ]
                Hotword list for character inputs = [ ['c', 'o', 'r', 'p'], ['t', 'e', 's', 't'] ]
        hotword_weight (Union[float, List[float]]) - Weight for each hotword. The weight for all the hotwords will be same when only one weight is provided.
            ( default = 10.0 )
        """
        if isinstance(hotword_weight, float) or isinstance(hotword_weight, int):
            hotword_weight = [hotword_weight] * len(hotwords)
        elif (
            isinstance(hotword_weight, List)
            and isinstance(hotwords, List)
            and len(hotwords) != len(hotword_weight)
        ):
            raise ValueError("Hotword weight list and Hotwords length doesn't match.")

        hotword_scorer = ctc_decode.get_hotword_scorer(
            self.decoder_options, hotwords, hotword_weight, self.token_separator
        )

        return hotword_scorer

    def decode(
        self,
        probs,
        seq_lens=None,
        hotword_scorer=None,
        hotwords: List[List[str]] = None,
        hotword_weight: Union[float, List[float]] = 10.0,
    ):
        """
        Conducts the beamsearch on model outputs and return results.
        Args:
        probs (Tensor) - A rank 3 tensor representing model outputs. Shape is batch x num_timesteps x num_labels.
        seq_lens (Tensor) - A rank 1 tensor representing the sequence length of the items in the batch. Optional,
        if not provided the size of axis 1 (num_timesteps) of `probs` is used for all items
        hotwords (List[List[str]]) - Tokenized list of hotwords.
            For example:
                Hotword list for BPE token inputs = [ ["co", "##r", "##p"], ["t", "##es","##t"] ]
                Hotword list for character inputs = [ ['c', 'o', 'r', 'p'], ['t', 'e', 's', 't'] ]
        hotword_weight (Union[float, List[float]]) - This is the boost factor for scoring the hotword when appeared in the beam path. Recommend to
            use the range between 0 - 15 for each hotword. If single value is provided then the same weigh will be used for all
            the hotwords

        Returns:
        tuple: (beam_results, beam_scores, timesteps, out_lens)

        beam_results (Tensor): A 3-dim tensor representing the top n beams of a batch of items.
                                Shape: batchsize x num_beams x num_timesteps.
                                Results are still encoded as ints at this stage.
        beam_scores (Tensor): A 3-dim tensor representing the likelihood of each beam in beam_results.
                                Shape: batchsize x num_beams x num_timesteps
        timesteps (Tensor): A 2-dim tensor representing the timesteps at which the nth output character
                                has peak probability.
                                To be used as alignment between audio and transcript.
                                Shape: batchsize x num_beams
        out_lens (Tensor): A 2-dim tensor representing the length of each beam in beam_results.
                                Shape: batchsize x n_beams.

        """
        probs = probs.cpu().float()
        batch_size, max_seq_len = probs.size(0), probs.size(1)
        if seq_lens is None:
            seq_lens = torch.IntTensor(batch_size).fill_(max_seq_len)
        else:
            seq_lens = seq_lens.cpu().int()

        if hotwords and hotword_scorer:
            raise ValueError(
                "You can only provide either a hotword or a hotword scorer, not both at the same time.\n"
            )

        # if hotwords list is provided then create a scorer for it
        if hotwords:
            hotword_scorer = self.create_hotword_scorer(hotwords, hotword_weight)

        output = torch.IntTensor(batch_size, self._beam_width, max_seq_len).cpu().int()
        timesteps = (
            torch.IntTensor(batch_size, self._beam_width, max_seq_len).cpu().int()
        )
        scores = torch.FloatTensor(batch_size, self._beam_width).cpu().float()
        out_seq_len = torch.zeros(batch_size, self._beam_width).cpu().int()

        if not self._scorer and not hotword_scorer:
            ctc_decode.paddle_beam_decode(
                probs,
                seq_lens,
                self.decoder_options,
                output,
                timesteps,
                scores,
                out_seq_len,
            )
        elif self._scorer and not hotword_scorer:
            ctc_decode.paddle_beam_decode_with_lm(
                probs,
                seq_lens,
                self.decoder_options,
                self._scorer,
                output,
                timesteps,
                scores,
                out_seq_len,
            )
        elif not self._scorer and hotword_scorer:
            ctc_decode.paddle_beam_decode_with_hotwords(
                probs,
                seq_lens,
                self.decoder_options,
                hotword_scorer,
                output,
                timesteps,
                scores,
                out_seq_len,
            )
        else:
            ctc_decode.paddle_beam_decode_with_lm_and_hotwords(
                probs,
                seq_lens,
                self.decoder_options,
                self._scorer,
                hotword_scorer,
                output,
                timesteps,
                scores,
                out_seq_len,
            )

        if hotwords:
            self.delete_hotword_scorer(hotword_scorer)

        return output, scores, timesteps, out_seq_len

    def character_based(self):
        return ctc_decode.is_character_based(self._scorer) if self._scorer else None

    def max_order(self):
        return ctc_decode.get_max_order(self._scorer) if self._scorer else None

    def dict_size(self):
        return ctc_decode.get_dict_size(self._scorer) if self._scorer else None

    def reset_params(self, alpha, beta):
        if self._scorer is not None:
            ctc_decode.reset_params(self._scorer, alpha, beta)

    def __del__(self):
        if self._scorer is not None:
            ctc_decode.paddle_release_scorer(self._scorer)
        if self.decoder_options:
            ctc_decode.paddle_release_decoder_options(self.decoder_options)

    def delete_hotword_scorer(self, hw_scorer=None):
        if hw_scorer:
            ctc_decode.paddle_release_hotword_scorer(hw_scorer)


class OnlineCTCBeamDecoder(object):
    """
    PyTorch wrapper for DeepSpeech PaddlePaddle Beam Search Decoder with interface for online decoding.
    Args:
        labels (list): The tokens/vocab used to train your model.
                        They should be in the same order as they are in your model's outputs.
        model_path (basestring): The path to your external KenLM language model(LM)
        alpha (float): Weighting associated with the LMs probabilities.
                        A weight of 0 means the LM has no effect.
        beta (float):  Weight associated with the number of words within our beam.
        cutoff_top_n (int): Cutoff number in pruning. Only the top cutoff_top_n characters
                            with the highest probability in the vocab will be used in beam search.
        cutoff_prob (float): Cutoff probability in pruning. 1.0 means no pruning.
        beam_width (int): This controls how broad the beam search is. Higher values are more likely to find top beams,
                            but they also will make your beam search exponentially slower.
        num_processes (int): Parallelize the batch using num_processes workers.
        blank_id (int): Index of the CTC blank token (probably 0) used when training your model.
        log_probs_input (bool): False if your model has passed through a softmax and output probabilities sum to 1.
        is_bpe_based (bool): True if your labels contains bpe tokens else False
        lm_type (str): Whether the language model file is character, bpe or word based
        token_separator (str): prefix of the bpe tokens. Default value is "#" and it is always assumed that the tokens
            starting with this prefix are meant to be merged with tokens that doesn't contain this prefix
        lexicon_fst_path (str): Path to the fst model file for decoding. It can be either be optimized or not. If not provided then
            fst will not be used for decoding. Default value is None.
    """

    def __init__(
        self,
        labels,
        model_path=None,
        alpha=0,
        beta=0,
        cutoff_top_n=40,
        cutoff_prob=1.0,
        beam_width=100,
        num_processes=4,
        blank_id=0,
        log_probs_input=False,
        is_bpe_based: bool = False,
        unk_score: float = -5.0,
        lm_type: str = "character",
        token_separator: str = "#",
        lexicon_fst_path: Optional[str] = None,
    ):
        self._cutoff_top_n = cutoff_top_n
        self._beam_width = beam_width
        self._scorer = None
        self._num_processes = num_processes
        self._labels = list(labels)  # Ensure labels are a list
        self._num_labels = len(labels)
        self._blank_id = blank_id
        self._log_probs = 1 if log_probs_input else 0
        lexicon_fst_path = lexicon_fst_path if lexicon_fst_path is not None else ""

        self.decoder_options = ctc_decode.paddle_get_decoder_options(
            self._labels,
            cutoff_top_n,
            cutoff_prob,
            beam_width,
            num_processes,
            blank_id,
            self._log_probs,
            is_bpe_based,
            unk_score,
            token_separator,
        )

        if model_path:
            self._scorer = ctc_decode.paddle_get_scorer(
                alpha,
                beta,
                model_path.encode(),
                self._labels,
                lm_type,
                lexicon_fst_path.encode(),
            )
        self._cutoff_prob = cutoff_prob

    def decode(self, probs, states, is_eos_s, seq_lens=None):
        """
        Conducts the beamsearch on model outputs and return results.
        Args:
        probs (Tensor) - A rank 3 tensor representing model outputs. Shape is batch x num_timesteps x num_labels.
        states (Sequence[DecoderState]) - sequence of decoding states with lens equal to batch_size.
        is_eos_s (Sequence[bool]) - sequence of bool with lens equal to batch size.
        Should have False if havent pushed all chunks yet, and True if you pushed last cank and you want to get an answer
        seq_lens (Tensor) - A rank 1 tensor representing the sequence length of the items in the batch. Optional,
        if not provided the size of axis 1 (num_timesteps) of `probs` is used for all items

        Returns:
        tuple: (beam_results, beam_scores, timesteps, out_lens)

        beam_results (Tensor): A 3-dim tensor representing the top n beams of a batch of items.
                                Shape: batchsize x num_beams x num_timesteps.
                                Results are still encoded as ints at this stage.
        beam_scores (Tensor): A 3-dim tensor representing the likelihood of each beam in beam_results.
                                Shape: batchsize x num_beams x num_timesteps
        timesteps (Tensor): A 2-dim tensor representing the timesteps at which the nth output character
                                has peak probability.
                                To be used as alignment between audio and transcript.
                                Shape: batchsize x num_beams
        out_lens (Tensor): A 2-dim tensor representing the length of each beam in beam_results.
                                Shape: batchsize x n_beams.

        """
        probs = probs.cpu().float()
        batch_size, max_seq_len = probs.size(0), probs.size(1)
        if seq_lens is None:
            seq_lens = torch.IntTensor(batch_size).fill_(max_seq_len)
        else:
            seq_lens = seq_lens.cpu().int()
        scores = torch.FloatTensor(batch_size, self._beam_width).cpu().float()
        out_seq_len = torch.zeros(batch_size, self._beam_width).cpu().int()

        decode_fn = ctc_decode.paddle_beam_decode_with_given_state
        res_beam_results, res_timesteps = decode_fn(
            probs,
            seq_lens,
            self._num_processes,
            [state.state for state in states],
            is_eos_s,
            scores,
            out_seq_len,
        )
        res_beam_results = res_beam_results.int()
        res_timesteps = res_timesteps.int()

        return res_beam_results, scores, res_timesteps, out_seq_len

    def character_based(self):
        return ctc_decode.is_character_based(self._scorer) if self._scorer else None

    def max_order(self):
        return ctc_decode.get_max_order(self._scorer) if self._scorer else None

    def dict_size(self):
        return ctc_decode.get_dict_size(self._scorer) if self._scorer else None

    def reset_state(state):
        ctc_decode.paddle_release_state(state)


class DecoderState:
    """
    Class using for maintain different chunks of data in one beam algorithm corresponding to one unique source.
    Note: after using State you should delete it, so dont reuse it
    Args:
        decoder (OnlineCTCBeamDecoder) - decoder you will use for decoding.
    """

    def __init__(self, decoder):
        self.state = ctc_decode.paddle_get_decoder_state(
            decoder.decoder_options,
            decoder._scorer,
        )

    def __del__(self):
        ctc_decode.paddle_release_state(self.state)
