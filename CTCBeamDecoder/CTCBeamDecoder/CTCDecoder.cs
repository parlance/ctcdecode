using System;
using System.Linq;
using CTCBeamDecoder.CTCBeamDecoderInterop;
using CTCBeamDecoder.Models;

namespace CTCBeamDecoder;

/// <summary>
/// C# wrapper for DeepSpeech PaddlePaddle Beam Search Decoder.
/// </summary>
public class CTCDecoder
{
    private readonly uint _cutoffTopN;
    private readonly double _cutoffProb;
    private readonly uint _beamWidth;
    private readonly uint _numProcesses;
    private readonly uint _blankId;
    private readonly int _logProbsInput;

    /// <param name="cutoffTopN">Cutoff number in pruning. Only the top cutoff_top_n characters with the highest probability in the vocab will be used in beam search.</param>
    /// <param name="cutoffProb">Cutoff probability in pruning. 1.0 means no pruning.</param>
    /// <param name="beamWidth">This controls how broad the beam search is. Higher values are more likely to find top beams, but they also will make your beam search exponentially slower.</param>
    /// <param name="numProcesses">Parallelize the batch using numProcesses workers.</param>
    /// <param name="blankId">Index of the CTC blank token (probably 0) used when training your model.</param>
    /// <param name="logProbsInput">false if your model has passed through a softmax and output probabilities sum to 1.</param>
    public CTCDecoder(uint cutoffTopN = 40,
                      double cutoffProb = 1,
                      uint beamWidth = 100,
                      uint numProcesses = 4,
                      uint blankId = 0,
                      bool logProbsInput = false)
    {
        _cutoffTopN = cutoffTopN;
        _cutoffProb = cutoffProb;
        _beamWidth = beamWidth;
        _numProcesses = numProcesses;
        _blankId = blankId;
        _logProbsInput = logProbsInput ? 1 : 0;
    }

    public DecoderState GetState(DecoderScorer scorer) =>
        new(CTCDecoderInterop.GetDecoderState(scorer.Labels, _beamWidth, _cutoffProb, _cutoffTopN, _blankId, _logProbsInput, scorer.Scorer));

    /// <summary>
    /// Conducts the beamsearch on model outputs and return results.
    /// </summary>
    /// <param name="probs"> A rank 3 tensor representing model outputs. Shape: BATCHSIZE x N_TIMESTEPS x N_LABELS.</param>
    /// <param name="scorer">Scorer to be used in decoding.</param>
    /// <param name="decoderResult">Result buffer. Note: This method cleans up before using decoderResult.</param>
    /// <param name="seqLens">Representing the sequence length of the items in the batch. Optional, if not provided the size of axis 1 (N_TIMESTEPS) of `probs` is used for all items</param>
    public void Decode(float[,,] probs, DecoderScorer scorer, DecoderResult decoderResult, int[]? seqLens = null) =>
        DecodeInternal(probs, scorer, decoderResult, true, seqLens);

    /// <summary>
    /// Conducts the beamsearch on model outputs and return results.
    /// </summary>
    /// <param name="probs"> A rank 3 tensor representing model outputs. Shape: BATCHSIZE x N_TIMESTEPS x N_LABELS.</param>
    /// <param name="scorer">Scorer to be used in decoding.</param>
    /// <param name="seqLens">Representing the sequence length of the items in the batch. Optional, if not provided the size of axis 1 (N_TIMESTEPS) of `probs` is used for all items</param>
    public DecoderResult Decode(float[,,] probs, DecoderScorer scorer, int[]? seqLens = null)
    {
        var batchSize = probs.GetLength(0);
        var timeStepsCount = probs.GetLength(1);

        var decoderResult = new DecoderResult(batchSize, _beamWidth, timeStepsCount);
        DecodeInternal(probs, scorer, decoderResult, false, seqLens);

        return decoderResult;
    }

    /// <summary>
    /// Conducts the beamsearch on model outputs and return results.
    /// </summary>
    /// <param name="probs"> A rank 3 tensor representing model outputs. Shape: BATCHSIZE x N_TIMESTEPS x N_LABELS.</param>
    /// <param name="scorer">Scorer to be used in decoding.</param>
    /// <param name="decoderResult">Result buffer. Note: This method cleans up before using decoderResult.</param>
    /// <param name="seqLens">Representing the sequence length of the items in the batch. Optional, if not provided the size of axis 1 (N_TIMESTEPS) of `probs` is used for all items</param>
    public void Decode(float[] probs, int batchSize, int timeStepsCount, int numClasses, DecoderScorer scorer, DecoderResultArray decoderResult, int[]? seqLens = null) =>
        DecodeInternal(probs, batchSize, timeStepsCount, numClasses, scorer, decoderResult, true, seqLens);

    /// <summary>
    /// Conducts the beamsearch on model outputs and return results.
    /// </summary>
    /// <param name="probs"> A rank 3 tensor representing model outputs. Shape: BATCHSIZE x N_TIMESTEPS x N_LABELS.</param>
    /// <param name="scorer">Scorer to be used in decoding.</param>
    /// <param name="seqLens">Representing the sequence length of the items in the batch. Optional, if not provided the size of axis 1 (N_TIMESTEPS) of `probs` is used for all items</param>
    public DecoderResultArray Decode(float[] probs, int batchSize, int timeStepsCount, int numClasses, DecoderScorer scorer, int[]? seqLens = null)
    {
        var decoderResult = new DecoderResultArray(batchSize, _beamWidth, timeStepsCount);
        DecodeInternal(probs, batchSize, timeStepsCount, numClasses, scorer, decoderResult, false, seqLens);

        return decoderResult;
    }

    private void DecodeInternal(float[,,] probs, DecoderScorer scorer, DecoderResult decoderResult, bool isNeedClean, int[]? seqLens = null)
    {
        var batchSize = probs.GetLength(0);
        var timeStepsCount = probs.GetLength(1);

        AssertDimensions(decoderResult, batchSize, timeStepsCount, seqLens);

        if (isNeedClean)
        {
            decoderResult.Clean();
        }

        seqLens ??= Enumerable.Repeat(timeStepsCount, batchSize).ToArray();

        if (!scorer.IsEmptyScorer)
        {
            CTCDecoderInterop.BeamDecodeLm(probs,
                                           seqLens,
                                           scorer.Labels,
                                           _beamWidth,
                                           _numProcesses,
                                           _cutoffProb,
                                           _cutoffTopN,
                                           _blankId,
                                           _logProbsInput,
                                           scorer.Scorer,
                                           decoderResult.BeamResults,
                                           decoderResult.TimeSteps,
                                           decoderResult.BeamScores,
                                           decoderResult.OutLens);
        }
        else
        {
            CTCDecoderInterop.BeamDecode(probs,
                                         seqLens,
                                         scorer.Labels,
                                         _beamWidth,
                                         _numProcesses,
                                         _cutoffProb,
                                         _cutoffTopN,
                                         _blankId,
                                         _logProbsInput,
                                         decoderResult.BeamResults,
                                         decoderResult.TimeSteps,
                                         decoderResult.BeamScores,
                                         decoderResult.OutLens);
        }
    }

    private void DecodeInternal(float[] probs, int batchSize, int timeStepsCount, int numClasses, DecoderScorer scorer, DecoderResultArray decoderResult, bool isNeedClean, int[]? seqLens = null)
    {
        AssertDimensions(decoderResult, batchSize, timeStepsCount, seqLens);

        if (isNeedClean)
        {
            decoderResult.Clean();
        }

        seqLens ??= Enumerable.Repeat(timeStepsCount, batchSize).ToArray();

        if (!scorer.IsEmptyScorer)
        {
            CTCDecoderInterop.BeamDecodeLm(probs,
                                           seqLens,
                                           scorer.Labels,
                                           (uint) batchSize,
                                           (uint) timeStepsCount,
                                           (uint) numClasses,
                                           _beamWidth,
                                           _numProcesses,
                                           _cutoffProb,
                                           _cutoffTopN,
                                           _blankId,
                                           _logProbsInput,
                                           scorer.Scorer,
                                           decoderResult.BeamResults,
                                           decoderResult.TimeSteps,
                                           decoderResult.BeamScores,
                                           decoderResult.OutLens);
        }
        else
        {
            CTCDecoderInterop.BeamDecode(probs,
                                         seqLens,
                                         scorer.Labels,
                                         (uint) batchSize,
                                         (uint) timeStepsCount,
                                         (uint) numClasses,
                                         _beamWidth,
                                         _numProcesses,
                                         _cutoffProb,
                                         _cutoffTopN,
                                         _blankId,
                                         _logProbsInput,
                                         decoderResult.BeamResults,
                                         decoderResult.TimeSteps,
                                         decoderResult.BeamScores,
                                         decoderResult.OutLens);
        }
    }

    /// <summary>
    /// Decode with given scorer into result.
    /// </summary>
    /// <param name="probs"> A rank 3 tensor representing model outputs. Shape: BATCHSIZE x N_TIMESTEPS x N_LABELS.</param>
    /// <param name="states">Sequence of decoding states. Shape: BATCHSIZE</param>
    /// <param name="isEosS">Sequence of bool with lens equal to batch size.Should have `false` if haven`t pushed all chunks yet, and True if you pushed last chunk and you want to get an answer</param>
    /// <param name="decoderResult">Result buffer. Note: This method cleans up before using decoderResult.</param>
    /// <param name="seqLens">Representing the sequence length of the items in the batch. Optional, if not provided the size of axis 1 (N_TIMESTEPS) of `probs` is used for all items</param>
    public void DecodeOnline(float[] probs, int batchSize, int timeStepsCount, int numClasses, DecoderState[] states, bool[] isEosS, DecoderResultArray decoderResult, int[]? seqLens = null) =>
        DecodeOnlineInternal(probs, batchSize, timeStepsCount, numClasses, states, isEosS, decoderResult, true, seqLens);

    /// <summary>
    /// Conducts the beamsearch on model outputs and return results.
    /// </summary>
    /// <param name="probs"> A rank 3 tensor representing model outputs. Shape: BATCHSIZE x N_TIMESTEPS x N_LABELS.</param>
    /// <param name="states">Sequence of decoding states. Shape: BATCHSIZE</param>
    /// <param name="isEosS">Sequence of bool with lens equal to batch size.Should have `false` if haven`t pushed all chunks yet, and True if you pushed last chunk and you want to get an answer</param>
    /// <param name="seqLens">Representing the sequence length of the items in the batch. Optional, if not provided the size of axis 1 (N_TIMESTEPS) of `probs` is used for all items</param>
    public DecoderResultArray DecodeOnline(float[] probs, int batchSize, int timeStepsCount, int numClasses, DecoderState[] states, bool[] isEosS, int[]? seqLens = null)
    {
        var decoderResult = new DecoderResultArray(batchSize, _beamWidth, timeStepsCount);
        DecodeOnlineInternal(probs, batchSize, timeStepsCount, numClasses, states, isEosS, decoderResult, false, seqLens);

        return decoderResult;
    }

    private void DecodeOnlineInternal(float[] probs, int batchSize, int timeStepsCount, int numClasses, DecoderState[] states, bool[] isEosS, DecoderResultArray decoderResult, bool isNeedClean, int[]? seqLens = null)
    {
        AssertDimensions(decoderResult, batchSize, timeStepsCount, seqLens);

        if (isNeedClean)
        {
            decoderResult.Clean();
        }

        seqLens ??= Enumerable.Repeat(timeStepsCount, batchSize).ToArray();

        CTCDecoderInterop.BeamDecodeWithGivenState(probs,
                                                   seqLens, (uint) batchSize, (uint) timeStepsCount, (uint) numClasses,
                                                   _beamWidth,
                                                   _numProcesses,
                                                   states.Select(x => x.State)
                                                         .ToArray(),
                                                   isEosS,
                                                   decoderResult.BeamScores,
                                                   decoderResult.OutLens,
                                                   decoderResult.BeamResults,
                                                   decoderResult.TimeSteps);
    }

    /// <summary>
    /// Decode with given scorer into result.
    /// </summary>
    /// <param name="probs"> A rank 3 tensor representing model outputs. Shape: BATCHSIZE x N_TIMESTEPS x N_LABELS.</param>
    /// <param name="states">Sequence of decoding states. Shape: BATCHSIZE</param>
    /// <param name="isEosS">Sequence of bool with lens equal to batch size.Should have `false` if haven`t pushed all chunks yet, and True if you pushed last chunk and you want to get an answer</param>
    /// <param name="decoderResult">Result buffer. Note: This method cleans up before using decoderResult.</param>
    /// <param name="seqLens">Representing the sequence length of the items in the batch. Optional, if not provided the size of axis 1 (N_TIMESTEPS) of `probs` is used for all items</param>
    public void DecodeOnline(float[,,] probs, DecoderState[] states, bool[] isEosS, DecoderResult decoderResult, int[]? seqLens = null) =>
        DecodeOnlineInternal(probs, states, isEosS, decoderResult, true, seqLens);

    /// <summary>
    /// Conducts the beamsearch on model outputs and return results.
    /// </summary>
    /// <param name="probs"> A rank 3 tensor representing model outputs. Shape: BATCHSIZE x N_TIMESTEPS x N_LABELS.</param>
    /// <param name="states">Sequence of decoding states. Shape: BATCHSIZE</param>
    /// <param name="isEosS">Sequence of bool with lens equal to batch size.Should have `false` if haven`t pushed all chunks yet, and True if you pushed last chunk and you want to get an answer</param>
    /// <param name="seqLens">Representing the sequence length of the items in the batch. Optional, if not provided the size of axis 1 (N_TIMESTEPS) of `probs` is used for all items</param>
    public DecoderResult DecodeOnline(float[,,] probs, DecoderState[] states, bool[] isEosS, int[]? seqLens = null)
    {
        var batchSize = probs.GetLength(0);
        var timeStepsCount = probs.GetLength(1);
        var decoderResult = new DecoderResult(batchSize, _beamWidth, timeStepsCount);
        DecodeOnlineInternal(probs, states, isEosS, decoderResult, false, seqLens);

        return decoderResult;
    }

    private void DecodeOnlineInternal(float[,,] probs, DecoderState[] states, bool[] isEosS, DecoderResult decoderResult, bool isNeedClean, int[]? seqLens = null)
    {
        var batchSize = probs.GetLength(0);
        var timeStepsCount = probs.GetLength(1);

        AssertDimensions(decoderResult, batchSize, timeStepsCount, seqLens);

        if (isNeedClean)
        {
            decoderResult.Clean();
        }

        seqLens ??= Enumerable.Repeat(timeStepsCount, batchSize).ToArray();

        CTCDecoderInterop.BeamDecodeWithGivenState(probs,
                                                   seqLens,
                                                   _beamWidth,
                                                   _numProcesses,
                                                   states.Select(x => x.State)
                                                         .ToArray(),
                                                   isEosS,
                                                   decoderResult.BeamScores,
                                                   decoderResult.OutLens,
                                                   decoderResult.BeamResults,
                                                   decoderResult.TimeSteps);
    }

    private void AssertDimensions(IDecoderResult decoderResult, int batchSize, int timeStepsCount, int[]? seqLens = null)
    {
        if (decoderResult.BatchSize != batchSize)
        {
            throw new ArgumentOutOfRangeException("BeamResults dimension 0 must be equal to probs dimensional 0. " +
                                                  $"Real: '{decoderResult.BatchSize}' " +
                                                  $"Must: '{batchSize}'");
        }

        if (decoderResult.BeamWidth != _beamWidth)
        {
            throw new ArgumentOutOfRangeException($"BeamResults dimension 1 must be equal to beam width. " +
                                                  $"Real: '{decoderResult.BeamWidth}' " +
                                                  $"Must: '{_beamWidth}'");
        }

        if (decoderResult.TimeStepsCount != timeStepsCount)
        {
            throw new ArgumentOutOfRangeException($"BeamResults dimension 2 must be equal to probs dimensional 1. " +
                                                  $"Real: '{decoderResult.TimeStepsCount}' " +
                                                  $"Must: '{timeStepsCount}'");
        }

        if (seqLens is not null && seqLens.Length != batchSize)
        {
            throw new ArgumentOutOfRangeException(nameof(seqLens),
                                                  $"SeqLens dimension must be equal to probs dimensional 0. " +
                                                  $"Real: '{seqLens.Length}' " +
                                                  $"Must: '{batchSize}'");
        }
    }
}
