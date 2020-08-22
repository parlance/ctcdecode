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
    """

    def __init__(self, labels, model_path=None, alpha=0, beta=0, cutoff_top_n=40, cutoff_prob=1.0, beam_width=100,
                 num_processes=4, blank_id=0, log_probs_input=False):
        self.cutoff_top_n = cutoff_top_n
        self._beam_width = beam_width
        self._scorer = None
        self._num_processes = num_processes
        self._labels = list(labels)  # Ensure labels are a list
        self._num_labels = len(labels)
        self._blank_id = blank_id
        self._log_probs = 1 if log_probs_input else 0
        if model_path:
            self._scorer = ctc_decode.paddle_get_scorer(alpha, beta, model_path.encode(), self._labels,
                                                        self._num_labels)
        self._cutoff_prob = cutoff_prob

    def decode(self, probs, seq_lens=None):
        """
        Conducts the beamsearch on model outputs and return results.
        Args:
        probs (Tensor) - A rank 3 tensor representing model outputs. Shape is batch x num_timesteps x num_labels.
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
        output = torch.IntTensor(batch_size, self._beam_width, max_seq_len).cpu().int()
        timesteps = torch.IntTensor(batch_size, self._beam_width, max_seq_len).cpu().int()
        scores = torch.FloatTensor(batch_size, self._beam_width).cpu().float()
        out_seq_len = torch.zeros(batch_size, self._beam_width).cpu().int()
        if self._scorer:
            ctc_decode.paddle_beam_decode_lm(probs, seq_lens, self._labels, self._num_labels, self._beam_width,
                                             self._num_processes, self._cutoff_prob, self.cutoff_top_n, self._blank_id,
                                             self._log_probs, self._scorer, output, timesteps, scores, out_seq_len)
        else:
            ctc_decode.paddle_beam_decode(probs, seq_lens, self._labels, self._num_labels, self._beam_width,
                                          self._num_processes,
                                          self._cutoff_prob, self.cutoff_top_n, self._blank_id, self._log_probs,
                                          output, timesteps, scores, out_seq_len)

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
