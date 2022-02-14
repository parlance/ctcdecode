using System;
using System.Runtime.InteropServices;

namespace CTCBeamDecoder.CTCBeamDecoderInterop;

internal static class CTCDecoderInterop
{
    private static readonly INativeCTCBeamDecoder Interop;

    static CTCDecoderInterop()
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            Interop = new WindowsNativeNativeCtcBeamBeamDecoderInterop();

            return;
        }

        if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
        {
            Interop = new LinuxNativeNativeCtcBeamBeamDecoderInterop();

            return;
        }

        throw new NotSupportedException("Unsupported operation system.");
    }

    public static unsafe int BeamDecode(
        float[] thProbs, //batch_size*max_time*num_classes
        int[] thSeqLens, //batch_size
        string[] labels, //num_labels
        uint batchsize,
        uint max_time,
        uint num_classes,
        uint beamSize,
        uint numProcesses,
        double cutoffProb,
        uint cutoffTopN,
        uint blankId,
        int logInput,
        int[] thOutput, //batch_size*beam_size*max_time
        int[] thTimesteps, //batch_size*beam_size*max_time
        float[] thScores, //batch_size*beam_size
        int[] thOutLength //batch_size*beam_size
    )
    {
        fixed (float* probs = thProbs, scores = thScores)
        {
            fixed (int* seqLens = thSeqLens, output = thOutput, timeSteps = thTimesteps, outLength = thOutLength)
            {
                return Interop.paddle_beam_decode_call(probs,
                                                       seqLens,
                                                       labels,
                                                       batchsize,
                                                       max_time,
                                                       num_classes,
                                                       beamSize,
                                                       numProcesses,
                                                       cutoffProb,
                                                       cutoffTopN,
                                                       blankId,
                                                       logInput,
                                                       output,
                                                       timeSteps,
                                                       scores,
                                                       outLength);
            }
        }
    }

    public static unsafe int BeamDecode(
        float[,,] thProbs, //batch_size*max_time*num_classes
        int[] thSeqLens, //batch_size
        string[] labels, //num_labels
        uint beamSize,
        uint numProcesses,
        double cutoffProb,
        uint cutoffTopN,
        uint blankId,
        int logInput,
        int[,,] thOutput, //batch_size*beam_size*max_time
        int[,,] thTimesteps, //batch_size*beam_size*max_time
        float[,] thScores, //batch_size*beam_size
        int[,] thOutLength //batch_size*beam_size
    )
    {
        fixed (float* probs = thProbs, scores = thScores)
        {
            fixed (int* seqLens = thSeqLens, output = thOutput, timeSteps = thTimesteps, outLength = thOutLength)
            {
                return Interop.paddle_beam_decode_call(probs,
                                                       seqLens,
                                                       labels,
                                                       (uint) thProbs.GetLength(0),
                                                       (uint) thProbs.GetLength(1),
                                                       (uint) thProbs.GetLength(2),
                                                       beamSize,
                                                       numProcesses,
                                                       cutoffProb,
                                                       cutoffTopN,
                                                       blankId,
                                                       logInput,
                                                       output,
                                                       timeSteps,
                                                       scores,
                                                       outLength);
            }
        }
    }

    public static unsafe int BeamDecodeLm(
        float[] thProbs, //batch_size*max_time*num_classes
        int[] thSeqLens, //batch_size
        string[] labels, //num_labels
        uint batchsize,
        uint max_time,
        uint num_classes,
        uint beamSize,
        uint numProcesses,
        double cutoffProb,
        uint cutoffTopN,
        uint blankId,
        int logInput,
        IntPtr scorer,
        int[] thOutput, //batch_size*beam_size*max_time
        int[] thTimesteps, //batch_size*beam_size*max_time
        float[] thScores, //batch_size*beam_size
        int[] thOutLength //batch_size*beam_size
    )
    {
        fixed (float* probs = thProbs, scores = thScores)
        {
            fixed (int* seqLens = thSeqLens, output = thOutput, timeSteps = thTimesteps, outLength = thOutLength)
            {
                return Interop.paddle_beam_decode_lm_call(probs, seqLens, labels, batchsize, max_time, num_classes, beamSize, numProcesses, cutoffProb, cutoffTopN, blankId, logInput, scorer.ToPointer(), output,
                                                          timeSteps,
                                                          scores, outLength);
            }
        }
    }

    public static unsafe int BeamDecodeLm(
        float[,,] thProbs, //batch_size*max_time*num_classes
        int[] thSeqLens, //batch_size
        string[] labels, //num_labels
        uint beamSize,
        uint numProcesses,
        double cutoffProb,
        uint cutoffTopN,
        uint blankId,
        int logInput,
        IntPtr scorer,
        int[,,] thOutput, //batch_size*beam_size*max_time
        int[,,] thTimesteps, //batch_size*beam_size*max_time
        float[,] thScores, //batch_size*beam_size
        int[,] thOutLength //batch_size*beam_size
    )
    {
        fixed (float* probs = thProbs, scores = thScores)
        {
            fixed (int* seqLens = thSeqLens, output = thOutput, timeSteps = thTimesteps, outLength = thOutLength)
            {
                return Interop.paddle_beam_decode_lm_call(probs, seqLens, labels, (uint) thProbs.GetLength(0), (uint) thProbs.GetLength(1), (uint) thProbs.GetLength(2), beamSize, numProcesses, cutoffProb, cutoffTopN, blankId, logInput, scorer.ToPointer(), output,
                                                          timeSteps,
                                                          scores, outLength);
            }
        }
    }

    public static unsafe IntPtr GetScorer(double alpha,
                                          double beta,
                                          string lmPath,
                                          string[] labels)
        => new IntPtr(Interop.paddle_get_scorer_call(alpha, beta, lmPath, labels, (uint) labels.Length));

    public static unsafe IntPtr GetDecoderState(string[] labels, //num_labels
                                                uint beamSize,
                                                double cutoffProb,
                                                uint cutoffTopN,
                                                uint blankId,
                                                int logInput,
                                                IntPtr scorer)
        => new IntPtr(Interop.paddle_get_decoder_state_call(labels, (uint) labels.Length, beamSize, cutoffProb, cutoffTopN, blankId, logInput, scorer.ToPointer()));

    public static unsafe void BeamDecodeWithGivenState(float[] thProbs, //batchsize*max_time*num_classes
                                                       int[] thSeqLens, //batchsize
                                                       uint batchsize,
                                                       uint max_time,
                                                       uint num_classes,
                                                       uint beamSize,
                                                       uint numProcesses,
                                                       IntPtr[] states, //batchsize
                                                       bool[] isEosS, //batchsize
                                                       float[] thScores, //batchsize, beam_size
                                                       int[] thOutLength, //batchsize, beam_size
                                                       int[] outputTokensTensor, //batchsize x beam_size*max_time
                                                       int[] outputTimestepsTensor //batchsize x beam_size*max_time
    )
    {
        fixed (float* probs = thProbs, scores = thScores)
        {
            fixed (int* seqLens = thSeqLens, outLength = thOutLength, outputTokens = outputTokensTensor, outputTimesteps = outputTimestepsTensor)
            {
                fixed (bool* eos = isEosS)
                {
                    var statesPointers = new void*[states.Length];

                    for (int i = 0; i < states.Length; i++)
                    {
                        statesPointers[i] = states[i].ToPointer();
                    }

                    fixed (void** statesPtr = statesPointers)
                    {
                        Interop.paddle_beam_decode_with_given_state_call(probs, seqLens, batchsize, max_time, num_classes, beamSize, numProcesses, statesPtr, eos, scores, outLength, outputTokens,
                                                                         outputTimesteps);
                    }
                }
            }
        }
    }

    public static unsafe void BeamDecodeWithGivenState(float[,,] thProbs, //batchsize*max_time*num_classes
                                                       int[] thSeqLens, //batchsize
                                                       uint beamSize,
                                                       uint numProcesses,
                                                       IntPtr[] states, //batchsize
                                                       bool[] isEosS, //batchsize
                                                       float[,] thScores, //batchsize, beam_size
                                                       int[,] thOutLength, //batchsize, beam_size
                                                       int[,,] outputTokensTensor, //batchsize x beam_size*max_time
                                                       int[,,] outputTimestepsTensor //batchsize x beam_size*max_time
    )
    {
        fixed (float* probs = thProbs, scores = thScores)
        {
            fixed (int* seqLens = thSeqLens, outLength = thOutLength, outputTokens = outputTokensTensor, outputTimesteps = outputTimestepsTensor)
            {
                fixed (bool* eos = isEosS)
                {
                    var statesPointers = new void*[states.Length];

                    for (int i = 0; i < states.Length; i++)
                    {
                        statesPointers[i] = states[i].ToPointer();
                    }

                    fixed (void** statesPtr = statesPointers)
                    {
                        Interop.paddle_beam_decode_with_given_state_call(probs, seqLens, (uint) thProbs.GetLength(0), (uint) thProbs.GetLength(1), (uint) thProbs.GetLength(2), beamSize, numProcesses, statesPtr, eos, scores, outLength, outputTokens,
                                                                         outputTimesteps);
                    }
                }
            }
        }
    }

    public static unsafe void ReleaseScorer(IntPtr scorer) => Interop.paddle_release_scorer_call(scorer.ToPointer());

    public static unsafe void ReleaseState(IntPtr scorer) => Interop.paddle_release_state_call(scorer.ToPointer());

    public static unsafe double GetLogCondProb(IntPtr scorer, string[] words) => Interop.get_log_cond_prob_call(scorer.ToPointer(), words, (uint) words.Length);

    public static unsafe double GetSentLogProb(IntPtr scorer, string[] words) => Interop.get_sent_log_prob_call(scorer.ToPointer(), words, (uint) words.Length);

    public static unsafe int IsCharacterBased(IntPtr scorer) => Interop.is_character_based_call(scorer.ToPointer());

    public static unsafe uint GetMaxOrder(IntPtr scorer) => Interop.get_max_order_call(scorer.ToPointer());

    public static unsafe uint GetDictionarySize(IntPtr scorer) => Interop.get_dict_size_call(scorer.ToPointer());

    public static unsafe void ResetParameters(IntPtr scorer, double alpha, double beta) => Interop.reset_params_call(scorer.ToPointer(), alpha, beta);
}
