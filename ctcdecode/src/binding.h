#pragma once
#define _cdecl
#ifdef __declspec(dllexport)
#define ADDExport __declspec (dllexport)
#else
#define ADDExport

#endif // !__declspec (dllexport)

extern "C" {
	ADDExport  int _cdecl paddle_beam_decode(float* th_probs, //batch_size*max_time*num_labels
		int* th_seq_lens, //batch_size
		char const* const* labels, //num_labels
		size_t batch_size,
		size_t max_time,
		size_t num_labels,
		size_t beam_size,
		size_t num_processes,
		double cutoff_prob,
		size_t cutoff_top_n,
		size_t blank_id,
		int log_input,
		int* th_output, //batch_size*beam_size*max_time
		int* th_timesteps, //batch_size*beam_size*max_time
		float* th_scores, //batch_size*beam_size
		int* th_out_length //batch_size*beam_size
	);


	ADDExport  int _cdecl paddle_beam_decode_lm(float* th_probs, //batch_size*max_time*num_labels
		int* th_seq_lens, //batch_size
		char const* const* labels, //num_labels
		size_t batch_size,
		size_t max_time,
		size_t num_labels,
		size_t beam_size,
		size_t num_processes,
		double cutoff_prob,
		size_t cutoff_top_n,
		size_t blank_id,
		int log_input,
		void* scorer,
		int* th_output, //batch_size*beam_size*max_time
		int* th_timesteps, //batch_size*beam_size*max_time
		float* th_scores, //batch_size*beam_size
		int* th_out_length //batch_size*beam_size
	);

	ADDExport  void* _cdecl paddle_get_scorer(double alpha,
		double beta,
		const char* lm_path,
		char const* const* labels,
		size_t labels_size);


	ADDExport void* _cdecl paddle_get_decoder_state(char const* const* vocabulary,
		size_t vocabulary_size,
		size_t beam_size,
		double cutoff_prob,
		size_t cutoff_top_n,
		size_t blank_id,
		int log_input,
		void* scorer);

	ADDExport  void _cdecl paddle_release_scorer(void* scorer);
	ADDExport void _cdecl paddle_release_state(void* state);
	ADDExport  void _cdecl paddle_beam_decode_with_given_state(float* th_probs, //batchsize*max_time*num_classes
		int* th_seq_lens, //batchsize
		size_t batch_size,
		size_t max_time,
		size_t num_classes,
		size_t beam_size,
		size_t num_processes,
		void** states, //batchsize
		bool* is_eos_s, //batchsize
		float* th_scores, //batchsize, beam_size
		int* th_out_length, //batchsize, beam_size
		int* output_tokens_tensor, //batchsize x beam_size*max_time
		int* output_timesteps_tensor //batchsize x beam_size*max_time
	);

	ADDExport  int _cdecl is_character_based(void* scorer);
	ADDExport  size_t _cdecl get_max_order(void* scorer);
	ADDExport  size_t _cdecl get_dict_size(void* scorer);
	ADDExport  void _cdecl reset_params(void* scorer, double alpha, double beta);
	ADDExport  double _cdecl get_log_cond_prob(void* scorer,
		char const* const* words, //vocabulary_size
		size_t words_size);

	ADDExport  double _cdecl get_sent_log_prob(void* scorer,
		char const* const* words, //vocabulary_size
		size_t words_size);
}
