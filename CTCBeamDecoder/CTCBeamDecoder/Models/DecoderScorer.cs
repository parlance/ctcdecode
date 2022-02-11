using System;
using CTCBeamDecoder.CTCBeamDecoderInterop;

namespace CTCBeamDecoder.Models;

/// <summary>
/// Class representing scorer for ctc decoder.
/// </summary>
public class DecoderScorer : IDisposable
{
    public readonly string[] Labels;
    public bool IsEmptyScorer { get; private set; }
    private readonly string? _modelPath;
    private readonly double _alpha;
    private readonly double _beta;
    internal IntPtr Scorer { get; private set; }
    private bool disposedValue;

    /// <param name="labels">The tokens/vocab used to train your model. They should be in the same order as they are in your model's outputs.</param>
    /// <param name="modelPath">The path to your external KenLM language model(LM)</param>
    /// <param name="alpha">Weighting associated with the LMs probabilities. A weight of 0 means the LM has no effect.</param>
    /// <param name="beta">Weight associated with the number of words within our beam.</param>
    public DecoderScorer(string[] labels,
                         string? modelPath = null,
                         double alpha = 0,
                         double beta = 0)
    {
        Labels = labels;
        _modelPath = modelPath;
        _alpha = alpha;
        _beta = beta;

        if (_modelPath is not null)
        {
            IsEmptyScorer = false;
            Scorer = CTCDecoderInterop.GetScorer(_alpha, _beta, _modelPath, Labels);
        }
        else
        {
            IsEmptyScorer = true;
            Scorer = IntPtr.Zero;
        }
    }

    public double? GetLogCondProb(string[] words)
    {
        if (!IsEmptyScorer)
        {
            return CTCDecoderInterop.GetLogCondProb(Scorer, words);
        }

        return null;
    }

    public double? GetSentLogProb(string[] words)
    {
        if (!IsEmptyScorer)
        {
            return CTCDecoderInterop.GetSentLogProb(Scorer, words);
        }

        return null;
    }

    public int? CharacterBased()
    {
        if (!IsEmptyScorer)
        {
            return CTCDecoderInterop.IsCharacterBased(Scorer);
        }

        return null;
    }

    public uint? MaxOrder()
    {
        if (!IsEmptyScorer)
        {
            return CTCDecoderInterop.GetMaxOrder(Scorer);
        }

        return null;
    }

    public uint? DictionarySize()
    {
        if (!IsEmptyScorer)
        {
            return CTCDecoderInterop.GetDictionarySize(Scorer);
        }

        return null;
    }

    public void ResetParameters(double alpha = 0, double beta = 0)
    {
        if (!IsEmptyScorer)
        {
            CTCDecoderInterop.ResetParameters(Scorer, alpha, beta);
        }
    }

    protected virtual void Dispose(bool disposing)
    {
        if (!disposedValue)
        {
            if (!IsEmptyScorer)
            {
                CTCDecoderInterop.ReleaseScorer(Scorer);
            }

            IsEmptyScorer = true;
            Scorer = IntPtr.Zero;
            disposedValue = true;
        }
    }

    ~DecoderScorer()
    {
        Dispose(disposing: false);
    }

    public void Dispose()
    {
        Dispose(disposing: true);
        GC.SuppressFinalize(this);
    }
}
