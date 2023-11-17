int paddle_beam_decode(THFloatTensor* th_probs,
                       THIntTensor* th_seq_lens,
                       void* decoder_options,
                       THIntTensor* th_output,
                       THIntTensor* th_timesteps,
                       THFloatTensor* th_scores,
                       THIntTensor* th_out_length);

int paddle_beam_decode_with_lm(THFloatTensor* th_probs,
                               THIntTensor* th_seq_lens,
                               void* decoder_options,
                               void* scorer,
                               THIntTensor* th_output,
                               THIntTensor* th_timesteps,
                               THFloatTensor* th_scores,
                               THIntTensor* th_out_length);

int paddle_beam_decode_with_hotwords(THFloatTensor* th_probs,
                                     THIntTensor* th_seq_lens,
                                     void* decoder_options,
                                     void* hotword_scorer,
                                     THIntTensor* th_output,
                                     THIntTensor* th_timesteps,
                                     THFloatTensor* th_scores,
                                     THIntTensor* th_out_length);

int paddle_beam_decode_with_lm_and_hotwords(THFloatTensor* th_probs,
                                            THIntTensor* th_seq_lens,
                                            void* decoder_options,
                                            void* scorer,
                                            void* hotword_scorer,
                                            THIntTensor* th_output,
                                            THIntTensor* th_timesteps,
                                            THFloatTensor* th_scores,
                                            THIntTensor* th_out_length);

void* paddle_get_decoder_options(std::vector<std::string> vocab,
                                 size_t cutoff_top_n,
                                 double cutoff_prob,
                                 size_t beam_width,
                                 size_t num_processes,
                                 size_t blank_id,
                                 bool log_probs_input,
                                 bool is_bpe_based,
                                 float unk_score,
                                 char token_separator);

void* paddle_get_scorer(double alpha,
                        double beta,
                        const char* lm_path,
                        std::vector<std::string> labels,
                        std::string lm_type,
                        const char* fst_path);

void* get_hotword_scorer(void* decoder_options,
                         std::vector<std::vector<std::string>> hotwords,
                         std::vector<float> hotword_weights,
                         char token_separator);

void* paddle_get_decoder_state(void* decoder_options, void* scorer);

void paddle_release_scorer(void* scorer);
void paddle_release_decoder_options(void* decoder_options);
void paddle_release_hotword_scorer(void* scorer);
void paddle_release_state(void* state);

int is_character_based(void* scorer);
size_t get_max_order(void* scorer);
size_t get_lexicon_size(void* scorer);
void reset_params(void* scorer, double alpha, double beta);
