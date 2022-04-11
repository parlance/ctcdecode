using System;
using System.Runtime.InteropServices;
using System.Security;

namespace CTCBeamDecoder.CTCBeamDecoderInterop;

[SuppressUnmanagedCodeSecurity]
internal static class CTCDecoderInterop
{
    private const string _lib = "NativeCTCBeamDecoder";

    static CTCDecoderInterop()
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows) || RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
        {
            return;
        }

        throw new NotSupportedException("Unsupported operation system.");
    }

    /// <summary/>
    /// <param name="thProbs">batch_size*max_time*num_classes</param>
    /// <param name="thSeqLens">batch_size</param>
    /// <param name="labels">num_labels</param>
    /// <param name="batchsize"></param>
    /// <param name="max_time"></param>
    /// <param name="num_classes"></param>
    /// <param name="beamSize"></param>
    /// <param name="numProcesses"></param>
    /// <param name="cutoffProb"></param>
    /// <param name="cutoffTopN"></param>
    /// <param name="blankId"></param>
    /// <param name="logInput"></param>
    /// <param name="thOutput">batch_size*beam_size*max_time</param>
    /// <param name="thTimesteps">batch_size*beam_size*max_time</param>
    /// <param name="thScores">batch_size*beam_size</param>
    /// <param name="thOutLength">batch_size*beam_size</param>
    /// <returns></returns>
    public static unsafe int BeamDecode(
        float[] thProbs,
        int[] thSeqLens,
        string[] labels,
        uint batchsize,
        uint max_time,
        uint num_classes,
        uint beamSize,
        uint numProcesses,
        double cutoffProb,
        uint cutoffTopN,
        uint blankId,
        int logInput,
        int[] thOutput,
        int[] thTimesteps,
        float[] thScores,
        int[] thOutLength
    )
    {
        fixed (float* probs = thProbs, scores = thScores)
        {
            fixed (int* seqLens = thSeqLens,
                   output = thOutput,
                   timeSteps = thTimesteps,
                   outLength = thOutLength)
            {
                return paddle_beam_decode(probs,
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

    /// <summary/>
    /// <param name="thProbs">batch_size*max_time*num_classes</param>
    /// <param name="thSeqLens">batch_size</param>
    /// <param name="labels">num_labels</param>
    /// <param name="beamSize"></param>
    /// <param name="numProcesses"></param>
    /// <param name="cutoffProb"></param>
    /// <param name="cutoffTopN"></param>
    /// <param name="blankId"></param>
    /// <param name="logInput"></param>
    /// <param name="thOutput">batch_size*beam_size*max_time</param>
    /// <param name="thTimesteps">batch_size*beam_size*max_time</param>
    /// <param name="thScores">batch_size*beam_size</param>
    /// <param name="thOutLength">batch_size*beam_size</param>
    /// <returns></returns>
    public static unsafe int BeamDecode(
        float[,,] thProbs,
        int[] thSeqLens,
        string[] labels,
        uint beamSize,
        uint numProcesses,
        double cutoffProb,
        uint cutoffTopN,
        uint blankId,
        int logInput,
        int[,,] thOutput,
        int[,,] thTimesteps,
        float[,] thScores,
        int[,] thOutLength
    )
    {
        fixed (float* probs = thProbs, scores = thScores)
        {
            fixed (int* seqLens = thSeqLens,
                   output = thOutput,
                   timeSteps = thTimesteps,
                   outLength = thOutLength)
            {
                return paddle_beam_decode(probs,
                                          seqLens,
                                          labels,
                                          (uint)thProbs.GetLength(0),
                                          (uint)thProbs.GetLength(1),
                                          (uint)thProbs.GetLength(2),
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

    /// <summary/>
    /// <param name="thProbs">batch_size*max_time*num_classes</param>
    /// <param name="thSeqLens">batch_size</param>
    /// <param name="labels">num_labels</param>
    /// <param name="batchsize"></param>
    /// <param name="max_time"></param>
    /// <param name="num_classes"></param>
    /// <param name="beamSize"></param>
    /// <param name="numProcesses"></param>
    /// <param name="cutoffProb"></param>
    /// <param name="cutoffTopN"></param>
    /// <param name="blankId"></param>
    /// <param name="logInput"></param>
    /// <param name="scorer"></param>
    /// <param name="thOutput">batch_size*beam_size*max_time</param>
    /// <param name="thTimesteps">batch_size*beam_size*max_time</param>
    /// <param name="thScores">batch_size*beam_size</param>
    /// <param name="thOutLength">batch_size*beam_size</param>
    /// <returns></returns>
    public static unsafe int BeamDecodeLm(
        float[] thProbs,
        int[] thSeqLens,
        string[] labels,
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
        int[] thOutput,
        int[] thTimesteps,
        float[] thScores,
        int[] thOutLength
    )
    {
        fixed (float* probs = thProbs, scores = thScores)
        {
            fixed (int* seqLens = thSeqLens,
                   output = thOutput,
                   timeSteps = thTimesteps,
                   outLength = thOutLength)
            {
                return paddle_beam_decode_lm(probs, seqLens,
                                             labels, batchsize,
                                             max_time, num_classes,
                                             beamSize, numProcesses,
                                             cutoffProb, cutoffTopN,
                                             blankId, logInput,
                                             scorer.ToPointer(), output,
                                             timeSteps, scores, outLength);
            }
        }
    }

    /// <summary/>
    /// <param name="thProbs">batch_size*max_time*num_classes</param>
    /// <param name="thSeqLens">batch_size</param>
    /// <param name="labels">num_labels</param>
    /// <param name="beamSize"></param>
    /// <param name="numProcesses"></param>
    /// <param name="cutoffProb"></param>
    /// <param name="cutoffTopN"></param>
    /// <param name="blankId"></param>
    /// <param name="logInput"></param>
    /// <param name="scorer"></param>
    /// <param name="thOutput">batch_size*beam_size*max_time</param>
    /// <param name="thTimesteps">batch_size*beam_size*max_time</param>
    /// <param name="thScores">batch_size*beam_size</param>
    /// <param name="thOutLength">batch_size*beam_size</param>
    /// <returns/>
    public static unsafe int BeamDecodeLm(
        float[,,] thProbs,
        int[] thSeqLens,
        string[] labels,
        uint beamSize,
        uint numProcesses,
        double cutoffProb,
        uint cutoffTopN,
        uint blankId,
        int logInput,
        IntPtr scorer,
        int[,,] thOutput,
        int[,,] thTimesteps,
        float[,] thScores,
        int[,] thOutLength
    )
    {
        fixed (float* probs = thProbs, scores = thScores)
        {
            fixed (int* seqLens = thSeqLens,
                   output = thOutput,
                   timeSteps = thTimesteps,
                   outLength = thOutLength)
            {
                return paddle_beam_decode_lm(probs, seqLens,
                                             labels, (uint)thProbs.GetLength(0),
                                             (uint)thProbs.GetLength(1), (uint)thProbs.GetLength(2),
                                             beamSize, numProcesses,
                                             cutoffProb, cutoffTopN,
                                             blankId, logInput,
                                             scorer.ToPointer(), output,
                                             timeSteps, scores,
                                             outLength);
            }
        }
    }

    public static unsafe IntPtr GetScorer(double alpha,
                                          double beta,
                                          string lmPath,
                                          string[] labels)
        => new IntPtr(paddle_get_scorer(alpha, beta, lmPath, labels, (uint)labels.Length));

    public static unsafe IntPtr GetDecoderState(string[] labels, //num_labels
                                                uint beamSize,
                                                double cutoffProb,
                                                uint cutoffTopN,
                                                uint blankId,
                                                int logInput,
                                                IntPtr scorer)
        => new IntPtr(paddle_get_decoder_state(labels, (uint)labels.Length,
                                               beamSize, cutoffProb,
                                               cutoffTopN, blankId,
                                               logInput, scorer.ToPointer()));

    /// <summary/>
    /// <param name="thProbs">batchsize*max_time*num_classes</param>
    /// <param name="thSeqLens">batchsize</param>
    /// <param name="batchsize"></param>
    /// <param name="max_time"></param>
    /// <param name="num_classes"></param>
    /// <param name="beamSize"></param>
    /// <param name="numProcesses"></param>
    /// <param name="states">batchsize</param>
    /// <param name="isEosS"></param>
    /// <param name="thScores">batchsize, beam_size</param>
    /// <param name="thOutLength">/batchsize, beam_size</param>
    /// <param name="outputTokensTensor">batchsize x beam_size*max_time</param>
    /// <param name="outputTimestepsTensor">batchsize x beam_size*max_time</param>
    public static unsafe void BeamDecodeWithGivenState(float[] thProbs,
                                                       int[] thSeqLens,
                                                       uint batchsize,
                                                       uint max_time,
                                                       uint num_classes,
                                                       uint beamSize,
                                                       uint numProcesses,
                                                       IntPtr[] states,
                                                       bool[] isEosS,
                                                       float[] thScores,
                                                       int[] thOutLength,
                                                       int[] outputTokensTensor,
                                                       int[] outputTimestepsTensor
    )
    {
        fixed (float* probs = thProbs, scores = thScores)
        {
            fixed (int* seqLens = thSeqLens,
                   outLength = thOutLength,
                   outputTokens = outputTokensTensor,
                   outputTimesteps = outputTimestepsTensor)
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
                        paddle_beam_decode_with_given_state(probs, seqLens,
                                                            batchsize, max_time,
                                                            num_classes, beamSize,
                                                            numProcesses, statesPtr,
                                                            eos, scores,
                                                            outLength, outputTokens,
                                                            outputTimesteps);
                    }
                }
            }
        }
    }

    /// <summary/>
    /// <param name="thProbs">batchsize*max_time*num_classes</param>
    /// <param name="thSeqLens">batchsize</param>
    /// <param name="beamSize"></param>
    /// <param name="numProcesses"></param>
    /// <param name="states">batchsize</param>
    /// <param name="isEosS">batchsize</param>
    /// <param name="thScores">batchsize, beam_size</param>
    /// <param name="thOutLength">batchsize, beam_size</param>
    /// <param name="outputTokensTensor">batchsize*beam_size*max_time</param>
    /// <param name="outputTimestepsTensor">batchsize*beam_size*max_time</param>
    public static unsafe void BeamDecodeWithGivenState(float[,,] thProbs,
                                                       int[] thSeqLens,
                                                       uint beamSize,
                                                       uint numProcesses,
                                                       IntPtr[] states,
                                                       bool[] isEosS,
                                                       float[,] thScores,
                                                       int[,] thOutLength,
                                                       int[,,] outputTokensTensor,
                                                       int[,,] outputTimestepsTensor
    )
    {
        fixed (float* probs = thProbs, scores = thScores)
        {
            fixed (int* seqLens = thSeqLens,
                   outLength = thOutLength,
                   outputTokens = outputTokensTensor,
                   outputTimesteps = outputTimestepsTensor)
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
                        paddle_beam_decode_with_given_state(probs, seqLens,
                                                            (uint)thProbs.GetLength(0),
                                                            (uint)thProbs.GetLength(1),
                                                            (uint)thProbs.GetLength(2),
                                                            beamSize, numProcesses,
                                                            statesPtr, eos,
                                                            scores, outLength,
                                                            outputTokens, outputTimesteps);
                    }
                }
            }
        }
    }

    public static unsafe void ReleaseScorer(IntPtr scorer)
        => paddle_release_scorer(scorer.ToPointer());

    public static unsafe void ReleaseState(IntPtr scorer)
        => paddle_release_state(scorer.ToPointer());

    public static unsafe double GetLogCondProb(IntPtr scorer, string[] words)
        => get_log_cond_prob(scorer.ToPointer(), words, (uint)words.Length);

    public static unsafe double GetSentLogProb(IntPtr scorer, string[] words)
        => get_sent_log_prob(scorer.ToPointer(), words, (uint)words.Length);

    public static unsafe int IsCharacterBased(IntPtr scorer)
        => is_character_based(scorer.ToPointer());

    public static unsafe uint GetMaxOrder(IntPtr scorer)
        => get_max_order(scorer.ToPointer());

    public static unsafe uint GetDictionarySize(IntPtr scorer)
        => get_dict_size(scorer.ToPointer());

    public static unsafe void ResetParameters(IntPtr scorer, double alpha, double beta)
        => reset_params(scorer.ToPointer(), alpha, beta);

    #region library calls

    [DllImport(_lib, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    private static extern unsafe int paddle_beam_decode(
        float* th_probs, //batch_size*max_time*num_classes
        int* th_seq_lens, //batch_size
        [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)]
        string[] labels, //num_labels
        uint batch_size,
        uint max_time,
        uint num_labels,
        uint beam_size,
        uint num_processes,
        double cutoff_prob,
        uint cutoff_top_n,
        uint blank_id,
        int log_input,
        int* th_output, //batch_size*beam_size*max_time
        int* th_timesteps, //batch_size*beam_size*max_time
        float* th_scores, //batch_size*beam_size
        int* th_out_length //batch_size*beam_size
    );

    [DllImport(_lib, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    private static extern unsafe int paddle_beam_decode_lm(
        float* th_probs, //batch_size*max_time*num_classes
        int* th_seq_lens, //batch_size
        [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)]
        string[] labels, //num_labels
        uint batch_size,
        uint max_time,
        uint num_labels,
        uint beam_size,
        uint num_processes,
        double cutoff_prob,
        uint cutoff_top_n,
        uint blank_id,
        int log_input,
        void* scorer,
        int* th_output, //batch_size*beam_size*max_time
        int* th_timesteps, //batch_size*beam_size*max_time
        float* th_scores, //batch_size*beam_size
        int* th_out_length //batch_size*beam_size
    );

    [DllImport(_lib, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    private static extern unsafe void* paddle_get_scorer(double alpha,
                                                         double beta,
                                                         [MarshalAs(UnmanagedType.LPStr)] string lm_path,
                                                         [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)]
                                                         string[] labels, //num_labels
                                                         uint labels_size);

    [DllImport(_lib, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    private static extern unsafe void* paddle_get_decoder_state([MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)] string[] vocabulary, //num_labels
                                                                uint vocabulary_size,
                                                                uint beam_size,
                                                                double cutoff_prob,
                                                                uint cutoff_top_n,
                                                                uint blank_id,
                                                                int log_input,
                                                                void* scorer);

    [DllImport(_lib, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    private static extern unsafe void paddle_beam_decode_with_given_state(float* th_probs, //batchsize*max_time*num_classes
                                                                          int* th_seq_lens, //batchsize
                                                                          uint batch_size,
                                                                          uint max_time,
                                                                          uint num_classes,
                                                                          uint beam_size,
                                                                          uint num_processes,
                                                                          void** states, //batchsize
                                                                          bool* is_eos_s, //batchsize
                                                                          float* th_scores, //batchsize, beam_size
                                                                          int* th_out_length, //batchsize, beam_size
                                                                          int* output_tokens_tensor, //batchsize x beam_size
                                                                          int* output_timesteps_tensor //batchsize x beam_size
    );

    [DllImport(_lib, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    private static extern unsafe double get_log_cond_prob(void* scorer,
                                                          [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)]
                                                          string[] words, //num_labels
                                                          uint words_size);

    [DllImport(_lib, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    private static extern unsafe double get_sent_log_prob(void* scorer,
                                                          [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)]
                                                          string[] words, //num_labels
                                                          uint words_size);

    [DllImport(_lib, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    private static extern unsafe void paddle_release_scorer(void* scorer);

    [DllImport(_lib, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    private static extern unsafe void paddle_release_state(void* state);

    [DllImport(_lib, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    private static extern unsafe int is_character_based(void* scorer);

    [DllImport(_lib, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    private static extern unsafe uint get_max_order(void* scorer);

    [DllImport(_lib, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    private static extern unsafe uint get_dict_size(void* scorer);

    [DllImport(_lib, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    private static extern unsafe void reset_params(void* scorer, double alpha, double beta);

    #endregion
}
