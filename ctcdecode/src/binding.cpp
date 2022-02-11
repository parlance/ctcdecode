#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include "scorer.h"
#include "ctc_beam_search_decoder.h"
#include "utf8.h"
#include "binding.h"

using namespace std;

void place_to_output(size_t max_time, size_t beam_size, int* th_output, int* th_timesteps, float* th_scores,
	int* th_out_length,
	const std::vector<std::vector<std::pair<double, Output>>>& batch_results)
{
	for (int b = 0; b < batch_results.size(); ++b)
	{
		std::vector<std::pair<double, Output>> results = batch_results[b];
		for (int p = 0; p < results.size(); ++p)
		{
			std::pair<double, Output> n_path_result = results[p];
			Output output = n_path_result.second;
			std::vector<int> output_tokens = output.tokens;
			std::vector<int> output_timesteps = output.timesteps;
			for (int t = 0; t < output_tokens.size(); ++t)
			{
				th_output[(b * beam_size * max_time + p * max_time + t)] = output_tokens[t]; // fill output tokens
				th_timesteps[(b * beam_size * max_time + p * max_time + t)] = output_timesteps[t];
			}
			th_scores[(b * beam_size + p)] = n_path_result.first;
			th_out_length[(b * beam_size + p)] = output_tokens.size();
		}
	}
}

std::vector<std::vector<std::vector<double>>> create_inputs(float* th_probs, int* th_seq_lens, size_t batch_size,
	size_t max_time, size_t num_labels)
{
	std::vector<std::vector<std::vector<double>>> inputs;
	for (int b = 0; b < batch_size; ++b)
	{
		// avoid a crash by ensuring that an erroneous seq_len doesn't have us try to access memory we shouldn't
		int seq_len = std::min(th_seq_lens[b], static_cast<int>(max_time));
		std::vector<std::vector<double>> temp(seq_len, std::vector<double>(num_labels));
		for (int t = 0; t < seq_len; ++t)
		{
			for (int n = 0; n < num_labels; ++n)
			{
				float val = th_probs[b * (seq_len)*num_labels + t * num_labels + n];
				temp[t][n] = val;
			}
		}
		inputs.push_back(temp);
	}
	return inputs;
}


int beam_decode(float* th_probs, //batch_size*max_time*num_labels
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
)
{
	Scorer* ext_scorer = NULL;
	if (scorer != NULL)
	{
		ext_scorer = static_cast<Scorer*>(scorer);
	}

	std::vector<std::vector<std::vector<double>>> inputs = create_inputs(
		th_probs, th_seq_lens, batch_size, max_time, num_labels);

	const vector<string> new_vocab_vec(labels, labels + num_labels);
	std::vector<std::vector<std::pair<double, Output>>> batch_results =
		ctc_beam_search_decoder_batch(inputs, new_vocab_vec, beam_size, num_processes, cutoff_prob, cutoff_top_n,
			blank_id, log_input, ext_scorer);

	place_to_output(max_time, beam_size, th_output, th_timesteps, th_scores, th_out_length, batch_results);
	return 1;
}

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
	)
	{
		return beam_decode(th_probs, th_seq_lens, labels, batch_size, max_time, num_labels, beam_size,
			num_processes,
			cutoff_prob, cutoff_top_n, blank_id, log_input, NULL, th_output, th_timesteps, th_scores,
			th_out_length);
	}

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
	)
	{
		return beam_decode(th_probs, th_seq_lens, labels, batch_size, max_time, num_labels, beam_size,
			num_processes,
			cutoff_prob, cutoff_top_n, blank_id, log_input, scorer, th_output, th_timesteps, th_scores,
			th_out_length);
	}


	ADDExport  void* _cdecl paddle_get_scorer(double alpha,
		double beta,
		const char* lm_path,
		char const* const* new_vocab, //vocab_size
		size_t vocab_size
	)
	{
		const vector<string> new_vocab_vec(new_vocab, new_vocab + vocab_size);

		Scorer* scorer = new Scorer(alpha, beta, lm_path, new_vocab_vec);
		return static_cast<void*>(scorer);
	}

	ADDExport  void _cdecl beam_decode_with_given_state(float* th_probs, //batchsize*max_time*num_classes
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
	)
	{
		std::vector<std::vector<std::vector<double>>> inputs = create_inputs(
			th_probs, th_seq_lens, batch_size, max_time, num_classes);
		vector<bool> is_eos_s_vec(is_eos_s, is_eos_s + batch_size * sizeof(bool));

		vector<void*> states_vec;
		for (int b = 0; b < batch_size; b++)
		{
			states_vec.push_back(states[b]);
		}

		std::vector<std::vector<std::pair<double, Output>>> batch_results =
			ctc_beam_search_decoder_batch_with_states(inputs, num_processes, states_vec, is_eos_s_vec);

		place_to_output(max_time, beam_size, output_tokens_tensor, output_timesteps_tensor, th_scores, th_out_length,
			batch_results);
	}

	ADDExport  void _cdecl paddle_beam_decode_with_given_state(float* th_probs, //batchsize*max_time*num_classes
		int* th_seq_lens, //batchsize
		size_t num_classes,
		size_t batch_size,
		size_t max_time,
		size_t beam_size,
		size_t num_processes,
		void** states, //batchsize
		bool* is_eos_s, //batchsize
		float* th_scores, //batchsize, beam_size
		int* th_out_length, //batchsize, beam_size
		int* th_output_tokens, //batchsize x beam_size
		int* th_output_timesteps //batchsize x beam_size
	)
	{
		return beam_decode_with_given_state(th_probs, th_seq_lens, num_classes, batch_size, max_time, beam_size,
			num_processes, states, is_eos_s, th_scores, th_out_length, th_output_tokens,
			th_output_timesteps);
	}


	ADDExport  void* _cdecl paddle_get_decoder_state(char const* const* vocabulary, //vocabulary_size
		size_t vocabulary_size,
		size_t beam_size,
		double cutoff_prob,
		size_t cutoff_top_n,
		size_t blank_id,
		int log_input,
		void* scorer)
	{
		// DecoderState state(vocabulary, beam_size, cutoff_prob, cutoff_top_n, blank_id, log_input, ext_scorer);
		Scorer* ext_scorer = NULL;
		if (scorer != NULL)
		{
			ext_scorer = static_cast<Scorer*>(scorer);
		}
		const vector<string> vocabulary_vec(vocabulary, vocabulary + vocabulary_size);

		DecoderState* state = new DecoderState(vocabulary_vec, beam_size, cutoff_prob, cutoff_top_n, blank_id, log_input,
			ext_scorer);

		return static_cast<void*>(state);
	}

	ADDExport  void _cdecl paddle_release_state(void* state)
	{
		delete static_cast<DecoderState*>(state);
	}

	ADDExport  void _cdecl paddle_release_scorer(void* scorer)
	{
		delete static_cast<Scorer*>(scorer);
	}

	ADDExport  int _cdecl is_character_based(void* scorer)
	{
		Scorer* ext_scorer = static_cast<Scorer*>(scorer);
		return ext_scorer->is_character_based();
	}

	ADDExport  size_t _cdecl get_max_order(void* scorer)
	{
		Scorer* ext_scorer = static_cast<Scorer*>(scorer);
		return ext_scorer->get_max_order();
	}

	ADDExport  size_t _cdecl get_dict_size(void* scorer)
	{
		Scorer* ext_scorer = static_cast<Scorer*>(scorer);
		return ext_scorer->get_dict_size();
	}

	ADDExport  double _cdecl get_log_cond_prob(void* scorer,
		char const* const* words, //vocabulary_size
		size_t words_size)
	{
		Scorer* ext_scorer = static_cast<Scorer*>(scorer);
		const vector<string> vocabulary_vec(words, words + words_size);
		return ext_scorer->get_log_cond_prob(vocabulary_vec);
	}

	ADDExport  double _cdecl get_sent_log_prob(void* scorer,
		char const* const* words, //vocabulary_size
		size_t words_size)
	{
		Scorer* ext_scorer = static_cast<Scorer*>(scorer);
		const vector<string> vocabulary_vec(words, words + words_size);
		return ext_scorer->get_sent_log_prob(vocabulary_vec);
	}

	ADDExport  void _cdecl reset_params(void* scorer, double alpha, double beta)
	{
		Scorer* ext_scorer = static_cast<Scorer*>(scorer);
		ext_scorer->reset_params(alpha, beta);
	}
}
