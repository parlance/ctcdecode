namespace CTCBeamDecoder.Models;

/// <summary>
/// Class containing the result of decode.
/// </summary>
public sealed class DecoderResultArray : IDecoderResult
{
    public int BatchSize { get; }
    public uint BeamWidth { get; }
    public int TimeStepsCount { get; }
    /// <summary>
    /// A 3-dim tensor representing the top n beams of a batch of items. Shape: BATCHSIZE x BEAMWIDTH x N_TIMESTEPS. Results are still encoded as int`s at this stage.
    /// </summary>
    public readonly int[] BeamResults;
    /// <summary>
    /// A 2-dim tensor representing the likelihood of each beam in beam_results. Shape: BATCHSIZE x BEAMWIDTH.
    /// </summary>
    public readonly float[] BeamScores;
    /// <summary>
    /// A 3-dim tensor representing the timesteps at which the nth output character has peak probability. To be used as alignment between audio and transcript. Shape: BATCHSIZE x BEAMWIDTH x N_TIMESTEPS.
    /// </summary>
    public readonly int[] TimeSteps;
    /// <summary>
    /// A 2-dim tensor representing the length of each beam in beam_results.Shape: BATCHSIZE x BEAMWIDTH.
    /// </summary>
    public readonly int[] OutLens;

    /// <summary>
    /// Result of decoding
    /// </summary>
    /// <param name="batchSize">Size of 1 dimension of passed probs</param>
    /// <param name="beamWidth"><see cref="T:CTCBeamDecoder.CTCDecoder"/> beam width</param>
    /// <param name="timeStepsCount">Size of 2 dimension of passed probs into decode method <see cref="T:CTCBeamDecoder.CTCDecoder"/></param>
    public DecoderResultArray(int batchSize, uint beamWidth, int timeStepsCount)
    {
        BatchSize = batchSize;
        BeamWidth = beamWidth;
        TimeStepsCount = timeStepsCount;
        BeamResults = new int[batchSize * beamWidth * timeStepsCount];
        TimeSteps = new int[batchSize * beamWidth * timeStepsCount];
        BeamScores = new float[batchSize * beamWidth];
        OutLens = new int[batchSize * beamWidth];
    }

    /// <summary>
    /// Method fills inner multidimensional arrays by Zero value.
    /// </summary>
    public void Clean()
    {
        for (int i = 0; i < BatchSize; i++)
        {
            for (int j = 0; j < BeamWidth; j++)
            {
                BeamScores[i * BeamWidth + j] = 0f;
                OutLens[i * BeamWidth + j] = 0;

                for (int k = 0; k < TimeStepsCount; k++)
                {
                    BeamResults[i * BeamWidth * TimeStepsCount + j * TimeStepsCount + k] = 0;
                    TimeSteps[i * BeamWidth * TimeStepsCount + j * TimeStepsCount + k] = 0;
                }
            }
        }
    }
}
