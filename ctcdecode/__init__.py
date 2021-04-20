import torch


def _init_extension():
    import os
    import importlib
    import torch

    # load the custom_op_library and register the custom ops
    lib_dir = os.path.dirname(__file__)
    loader_details = (
        importlib.machinery.ExtensionFileLoader,
        importlib.machinery.EXTENSION_SUFFIXES
    )

    extfinder = importlib.machinery.FileFinder(lib_dir, loader_details)
    ext_specs = extfinder.find_spec("libctcdecode")
    if ext_specs is None:
        raise ImportError("ctc_decode C++ Extension is not found.")
    torch.ops.load_library(ext_specs.origin)
    torch.classes.load_library(ext_specs.origin)


_init_extension()


del _init_extension


class CTCBeamDecoder(object):
    """
    PyTorch wrapper for DeepSpeech PaddlePaddle Beam Search Decoder.
    Args:
        labels (list): The tokens/vocab used to train your model.
                        They should be in the same order as they are in your model's outputs.
        beam_width (int): This controls how broad the beam search is. Higher values are more likely to find top beams,
                            but they also will make your beam search exponentially slower.
        model_path (basestring): The path to your external KenLM language model(LM)
        alpha (float): Weighting associated with the LMs probabilities.
                        A weight of 0 means the LM has no effect.
        beta (float):  Weight associated with the number of words within our beam.
        cutoff_top_n (int): Cutoff number in pruning. Only the top cutoff_top_n characters
                            with the highest probability in the vocab will be used in beam search.
        cutoff_prob (float): Cutoff probability in pruning. 1.0 means no pruning.
        num_processes (int): Parallelize the batch using num_processes workers. 
        blank_id (int): Index of the CTC blank token (probably 0) used when training your model.
        log_probs_input (bool): False if your model has passed through a softmax and output probabilities sum to 1.
    """

    def __init__(self, labels, beam_width=100, cutoff_top_n=40, cutoff_prob=1.0,
                 num_processes=4, blank_id=0, log_probs_input=False):
        self.cutoff_top_n = cutoff_top_n
        self._beam_width = beam_width
        self._num_processes = num_processes
        self._labels = list(labels)  # Ensure labels are a list
        self._num_labels = len(labels)
        self._blank_id = blank_id
        self._log_probs = 1 if log_probs_input else 0
        self._cutoff_prob = cutoff_prob

    def decode(self, probs, seq_lens=None):
        """
        Conducts the beamsearch on model outputs and return results.
        Args:
            probs (torch.Tensor):
                Encoder output.
                Shape: ``[batch, num_timesteps, num_labels]``.
            seq_lens (torch.Tensor, optional):
                The sequence length of the items in the batch.
                Shape: ``[batch]``.
                If not provided, the size of axis 1 (``num_timesteps``) of ``probs``
                is used for all items.

        Returns:
            Tuple of four torch.Tensors: Tuple of ``beams``, ``length``, ``scores`` and ``timesteps``
            beams:
                Integer Tensor representing the top ``n`` beams.
                Shape: ``[batch, num_beams, num_timesteps]``.
            length:
                Integer Tensor representing the length of each beam.
                Shape: ``[batch, num_beams]``.
            scores:
                Float Tensor representing the likelihood of each beam.
                Shape: ``[batch, num_beams]``.
            timesteps:
                Integer Tensor representing the timesteps at which
                the corresponding output character has peak probability.
                Shape: ``[batch, num_beams, num_timesteps]``.
        """
        return torch.ops.ctcdecode.beam_decode(
            probs, seq_lens, self._labels, self._num_labels, self._beam_width,
            self._num_processes, self._cutoff_prob, self.cutoff_top_n,
            self._blank_id, self._log_probs)
