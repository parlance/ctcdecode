namespace CTCBeamDecoder.CTCBeamDecoderInterop;

internal interface INativeCTCBeamDecoder
{
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
    );

    public unsafe void* paddle_get_scorer_call(double alpha,
                                               double beta,
                                               string lm_path,
                                               string[] labels, //num_labels
                                               uint labels_size);

    public unsafe void* paddle_get_decoder_state_call(string[] vocabulary, //num_labels
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
    );

    public unsafe double get_log_cond_prob_call(void* scorer, string[] words, //num_labels
                                                uint words_size);

    public unsafe double get_sent_log_prob_call(void* scorer, string[] words, //num_labels
                                                uint words_size);

    public unsafe void paddle_release_scorer_call(void* scorer);

    public unsafe void paddle_release_state_call(void* state);

    public unsafe int is_character_based_call(void* scorer);

    public unsafe uint get_max_order_call(void* scorer);

    public unsafe uint get_dict_size_call(void* scorer);

    public unsafe void reset_params_call(void* scorer, double alpha, double beta);
}
