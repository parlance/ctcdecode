# Connectionist Temporal Classification(CTC) beam search decoding for C#(ctcdecode)
- [![lisence](https://img.shields.io/badge/lisence-MIT-green?style=flat-square)](https://github.com/aleksandr-aleksashin/ctcdecode/blob/master/LICENSE)
- [![nuget](https://img.shields.io/nuget/v/CTCBeamDecoder)](https://www.nuget.org/packages/CTCBeamDecoder)
- [![downloads](https://img.shields.io/nuget/dt/CTCBeamDecoder)](https://www.nuget.org/packages/CTCBeamDecoder)

ctcdecode is an implementation of CTC (Connectionist Temporal Classification) beam search decoding for C#.
C++ code borrowed liberally from `parlance ctcdecode` [parlance](https://github.com/parlance/ctcdecode).
It includes swappable scorer support enabling standard beam search, and KenLM-based decoding.   
If you are new to the concepts of CTC and Beam Search, please visit the Resources section where we link a few tutorials explaining why they are needed.  
**Attention**: Works only under `Linux` and `Windows`, because uses native libraries.  
## Installation
Building the C++ library requires gcc or clang. 
KenLM language modeling support is also optionally included, and enabled by default.

```bash
# get the code
git clone --recursive https://github.com/aleksandr-aleksashin/ctcdecode.git
cd ctcdecode
```
## Compilation
To compile native library needs to switch into directory `./ctcdecode`.
- For building Linux native library required `g++`. Run `make clean` and then `make`. This way works in WSL too.
- For building Windows native library required `MinGW64`. Run `make clean` and then `make`.

## How to Use

```c#
using CTCBeamDecoder;
using CTCBeamDecoder.Models;

using var scorer = new DecoderScorer(labels);
var decoder = new CTCDecoder();
var result = decoder.Decode(output, scorer);
```

### Inputs to `CTCDecoder`
 - `cutoffTopN` Cutoff number in pruning. Only the top cutoffTopN characters with the highest probability in the vocab will be used in beam search. Default `40`.  
 - `cutoffProb` Cutoff probability in pruning. 1.0 means no pruning. Default `1`.  
 - `beamWidth` This controls how broad the beam search is. Higher values are more likely to find top beams, but they also will make your beam search exponentially slower. Furthermore, the longer your outputs, the more time large beams will take. This is an important parameter that represents a tradeoff you need to make based on your dataset and needs. Default `100`.  
 - `numProcesses` Parallelize the batch using num_processes workers. You probably want to pass the number of cpus your computer has. Default `4`.  
 - `blankId` This should be the index of the CTC blank token (probably 0). Default `0`.  
 - `logProbsInput` If your outputs have passed through a softmax and represent probabilities, this should be false, if they passed through a LogSoftmax and represent negative log likelihood, you need to pass True. Default `false`.  

### Inputs to the `Decode` method
 - `output` should be the output activations from your model. If your output has passed through a SoftMax layer, you shouldn't need to alter it (except maybe to transpose), but if your `output` represents negative log likelihoods (raw logits), you either need to pass it through an additional `softmax` or you can pass `logProbsInput=False` to the decoder. Shape: BATCHSIZE x N_TIMESTEPS x N_LABELS.  
 - `scorer` to be used in decoding. 
 - `seqLens` representing the sequence length of the items in the batch. Shape: BATCHSIZE.  

### Inputs to the `OnlineDecode` method
 - `output` should be the output activations from your model. If your output has passed through a SoftMax layer, you shouldn't need to alter it (except maybe to transpose), but if your `output` represents negative log likelihoods (raw logits), you either need to pass it through an additional `softmax` or you can pass `logProbsInput=False` to the decoder. Shape: BATCHSIZE x N_TIMESTEPS x N_LABELS.  
 - `states` sequence of decoding states with lens equal to batch_size. Shape: BATCHSIZE.  
 - `isEosS` sequence of bool with lens equal to batch size. Should have False if havent pushed all chunks yet, and True if you pushed last cank and you want to get an answer. Shape: BATCHSIZE.  
 - `seqLens` representing the sequence length of the items in the batch. Shape: BATCHSIZE.  

### Inputs to `DecoderScorer`
 - `labels` are the tokens you used to train your model. They should be in the same order as your outputs. For example if your tokens are the english letters and you used 0 as your blank token, then you would pass in list "_abcdefghijklmopqrstuvwxyz") as your argument to labels.  
 - `modelPath` is the path to your external kenlm language model(LM). Default `null`.  
 - `alpha` Weighting associated with the LMs probabilities. A weight of 0 means the LM has no effect. Default `0`.  
 - `beta` Weight associated with the number of words within our beam. Default `0`.  

### Inputs to `DecoderState`
 - `scorer` associated scorer.

## Resources

- [Distill Guide to CTC](https://distill.pub/2017/ctc/)
- [Beam Search Video by Andrew Ng](https://www.youtube.com/watch?v=RLWuzLLSIgw)
- [An Intuitive Explanation of Beam Search](https://towardsdatascience.com/an-intuitive-explanation-of-beam-search-9b1d744e7a0f)

## License

Authored by: Aleksashin Aleksandr (aleksandr-aleksashin)

This project is under MIT license. You can obtain the license copy [here](https://github.com/aleksandr-aleksashin/ctcdecode/blob/master/LICENSE).