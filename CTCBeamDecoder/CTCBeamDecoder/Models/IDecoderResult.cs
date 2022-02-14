namespace CTCBeamDecoder.Models;

public interface IDecoderResult
{
    public int BatchSize { get; }
    public uint BeamWidth { get; }
    public int TimeStepsCount { get; }

    /// <summary>
    /// Method fills inner multidimensional arrays by Zero value.
    /// </summary>
    void Clean();
}