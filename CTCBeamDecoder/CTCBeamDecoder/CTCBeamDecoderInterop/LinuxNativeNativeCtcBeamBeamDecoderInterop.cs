using System.Runtime.InteropServices;

namespace CTCBeamDecoder.CTCBeamDecoderInterop;

internal sealed class LinuxNativeNativeCtcBeamBeamDecoderInterop : INativeCTCBeamDecoder
{
    private const string _lib = "NativeCTCBeamDecoder";

    public unsafe int paddle_beam_decode_call(
        float* th_probs, //batch_size*max_time*num_classes
        int* th_seq_lens, //batch_size
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
    ) => paddle_beam_decode(th_probs, th_seq_lens, labels, batch_size, max_time, num_labels, beam_size, num_processes, cutoff_prob, cutoff_top_n, blank_id, log_input, th_output, th_timesteps, th_scores, th_out_length);

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

    public unsafe int paddle_beam_decode_lm_call(
        float* th_probs, //batch_size*max_time*num_classes
        int* th_seq_lens, //batch_size
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
    ) => paddle_beam_decode_lm(th_probs, th_seq_lens, labels, batch_size, max_time, num_labels, beam_size, num_processes, cutoff_prob, cutoff_top_n, blank_id, log_input, scorer, th_output, th_timesteps, th_scores, th_out_length);

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

    public unsafe void* paddle_get_scorer_call(double alpha,
                                               double beta,
                                               string lm_path,
                                               string[] labels, //num_labels
                                               uint labels_size) => paddle_get_scorer(alpha, beta, lm_path, labels, labels_size);

    [DllImport(_lib, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    private static extern unsafe void* paddle_get_scorer(double alpha,
                                                         double beta,
                                                         [MarshalAs(UnmanagedType.LPStr)] string lm_path,
                                                         [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)]
                                                         string[] labels, //num_labels
                                                         uint labels_size);

    public unsafe void* paddle_get_decoder_state_call(string[] vocabulary, //num_labels
                                                      uint vocabulary_size,
                                                      uint beam_size,
                                                      double cutoff_prob,
                                                      uint cutoff_top_n,
                                                      uint blank_id,
                                                      int log_input,
                                                      void* scorer) => paddle_get_decoder_state(vocabulary, vocabulary_size, beam_size, cutoff_prob, cutoff_top_n, blank_id, log_input, scorer);

    [DllImport(_lib, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    private static extern unsafe void* paddle_get_decoder_state([MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)] string[] vocabulary, //num_labels
                                                                uint vocabulary_size,
                                                                uint beam_size,
                                                                double cutoff_prob,
                                                                uint cutoff_top_n,
                                                                uint blank_id,
                                                                int log_input,
                                                                void* scorer);

    public unsafe void paddle_beam_decode_with_given_state_call(float* th_probs, //batchsize*max_time*num_classes
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
    ) => paddle_beam_decode_with_given_state(th_probs, th_seq_lens, batch_size, max_time, num_classes, beam_size, num_processes, states, is_eos_s, th_scores, th_out_length, output_tokens_tensor, output_timesteps_tensor);

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

    public unsafe double get_log_cond_prob_call(void* scorer, string[] words, //num_labels
                                                uint words_size) => get_log_cond_prob(scorer, words, words_size);

    [DllImport(_lib, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    private static extern unsafe double get_log_cond_prob(void* scorer, [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)] string[] words, //num_labels
                                                          uint words_size);

    public unsafe double get_sent_log_prob_call(void* scorer, string[] words, //num_labels
                                                uint words_size) => get_sent_log_prob(scorer, words, words_size);

    [DllImport(_lib, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    private static extern unsafe double get_sent_log_prob(void* scorer, [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)] string[] words, //num_labels
                                                          uint words_size);

    public unsafe void paddle_release_scorer_call(void* scorer) => paddle_release_scorer(scorer);

    [DllImport(_lib, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    private static extern unsafe void paddle_release_scorer(void* scorer);

    public unsafe void paddle_release_state_call(void* state) => paddle_release_state(state);

    [DllImport(_lib, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    private static extern unsafe void paddle_release_state(void* state);

    public unsafe int is_character_based_call(void* scorer) => is_character_based(scorer);

    [DllImport(_lib, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    private static extern unsafe int is_character_based(void* scorer);

    public unsafe uint get_max_order_call(void* scorer) => get_max_order(scorer);

    [DllImport(_lib, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    private static extern unsafe uint get_max_order(void* scorer);

    public unsafe uint get_dict_size_call(void* scorer) => get_dict_size(scorer);

    [DllImport(_lib, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    private static extern unsafe uint get_dict_size(void* scorer);

    public unsafe void reset_params_call(void* scorer, double alpha, double beta) => reset_params(scorer, alpha, beta);

    [DllImport(_lib, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    private static extern unsafe void reset_params(void* scorer, double alpha, double beta);
}
