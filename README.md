# ctcdecode

ctcdecode is an implementation of CTC (Connectionist Temporal Classification) beam search decoding for PyTorch.
C++ code borrowed liberally from Paddle Paddles' [DeepSpeech](https://github.com/PaddlePaddle/DeepSpeech).
It includes swappable scorer support enabling standard beam search, and KenLM-based decoding. If you are new to the concepts of CTC and Beam Search, please visit the Resources section where we link a few tutorials explaining why they are needed. 

## Installation
The library is largely self-contained and requires only PyTorch. 
Building the C++ library requires gcc or clang. 
KenLM language modeling support is also optionally included, and enabled by default.

The below installation also works for Google Colab.

```bash
# get the code
git clone --recursive https://github.com/parlance/ctcdecode.git
cd ctcdecode && pip install .
```

## How to Use

```python
from ctcdecode import CTCBeamDecoder

decoder = CTCBeamDecoder(
    labels,
    model_path=None,
    alpha=0,
    beta=0,
    cutoff_top_n=40,
    cutoff_prob=1.0,
    beam_width=100,
    num_processes=4,
    blank_id=0,
    log_probs_input=False
)
beam_results, beam_scores, timesteps, out_lens = decoder.decode(output)
```

### Inputs to `CTCBeamDecoder`
 - `labels` are the tokens you used to train your model. They should be in the same order as your outputs. For example
 if your tokens are the english letters and you used 0 as your blank token, then you would
 pass in list("_abcdefghijklmopqrstuvwxyz") as your argument to labels
 - `model_path` is the path to your external kenlm language model(LM). Default is none.
 - `alpha` Weighting associated with the LMs probabilities. A weight of 0 means the LM has no effect.
 - `beta` Weight associated with the number of words within our beam.
 - `cutoff_top_n` Cutoff number in pruning. Only the top cutoff_top_n characters with the highest probability in the vocab will be used in beam search.
 - `cutoff_prob` Cutoff probability in pruning. 1.0 means no pruning.
 - `beam_width` This controls how broad the beam search is. Higher values are more likely to find top beams, but they also
 will make your beam search exponentially slower. Furthermore, the longer your outputs, the more time large beams will take.
  This is an important parameter that represents a tradeoff you need to make based on your dataset and needs.
 - `num_processes` Parallelize the batch using num_processes workers. You probably want to pass the number of cpus your computer has. You can find this in python with `import multiprocessing` then `n_cpus = multiprocessing.cpu_count()`. Default 4.
 - `blank_id` This should be the index of the CTC blank token (probably 0). 
 - `log_probs_input` If your outputs have passed through a softmax and represent probabilities, this should be false, if they passed through a LogSoftmax and represent negative log likelihood, you need to pass True. If you don't understand this, run `print(output[0][0].sum())`, if it's a negative number you've probably got NLL and need to pass True, if it sums to ~1.0 you should pass False. Default False.

### Inputs to the `decode` method
 - `output` should be the output activations from your model. If your output has passed through a SoftMax layer, you shouldn't need to alter it (except maybe to transpose), but if your `output` represents negative log likelihoods (raw logits), you either need to pass it through an additional `torch.nn.functional.softmax` or you can pass `log_probs_input=False` to the decoder. Your output should be BATCHSIZE x N_TIMESTEPS x N_LABELS so you may need to transpose it before passing it to the decoder. Note that if you pass things in the wrong order, the beam search will probably still run, you'll just get back nonsense results. 

### Outputs from the `decode` method

4 things get returned from `decode`
 1. `beam_results` - Shape: BATCHSIZE x N_BEAMS X N_TIMESTEPS A batch containing the series of characters (these are ints, you still need to decode them back to your text) representing results from a given beam search. Note that the beams are almost always shorter than the total number of timesteps, and the additional data is non-sensical, so to see the top beam (as int labels) from the first item in the batch, you need to run `beam_results[0][0][:out_len[0][0]]`.
 1. `beam_scores` - Shape: BATCHSIZE x N_BEAMS A batch with the approximate CTC score of each beam (look at the code [here](https://github.com/parlance/ctcdecode/blob/master/ctcdecode/src/ctc_beam_search_decoder.cpp#L191-L192) for more info). If this is true, you can get the model's confidence that the beam is correct with `p=1/np.exp(beam_score)`.
 1. `timesteps` - Shape: BATCHSIZE x N_BEAMS The timestep at which the nth output character has peak probability. Can be used as alignment between the audio and the transcript.
 1. `out_lens` - Shape: BATCHSIZE x N_BEAMS. `out_lens[i][j]` is the length of the jth beam_result, of item i of your batch. 

 ### More examples

Get the top beam for the first item in your batch
`beam_results[0][0][:out_len[0][0]]`

Get the top 50 beams for the first item in your batch
```python
for i in range(50):
     print(beam_results[0][i][:out_len[0][i]])
```

Note, these will be a list of ints that need decoding. You likely already have a function to decode from int to text, but if not you can do something like.
`"".join[labels[n] for n in beam_results[0][0][:out_len[0][0]]]` using the labels you passed in to `CTCBeamDecoder`


## Resources

- [Distill Guide to CTC](https://distill.pub/2017/ctc/)
- [Beam Search Video by Andrew Ng](https://www.youtube.com/watch?v=RLWuzLLSIgw)
- [An Intuitive Explanation of Beam Search](https://towardsdatascience.com/an-intuitive-explanation-of-beam-search-9b1d744e7a0f)
