#include "boost/python.hpp"
#include "boost/python/stl_iterator.hpp"
#include "boost/shared_ptr.hpp"
#include <pybind11/pybind11.h>
#include <torch/torch.h>

#include "ctc_beam_search_decoder.h"
#include "decoder_options.h"
#include "scorer.h"
#include "utf8.h"

namespace py = pybind11;

template <typename T>
inline std::vector<T> py_list_to_std_vector(const boost::python::object& iterable)
{
    return std::vector<T>(boost::python::stl_input_iterator<T>(iterable),
                          boost::python::stl_input_iterator<T>());
}

template <class T>
inline boost::python::list std_vector_to_py_list(std::vector<T> vector)
{
    typename std::vector<T>::iterator iter;
    boost::python::list list;
    for (iter = vector.begin(); iter != vector.end(); ++iter) {
        list.append(*iter);
    }
    return list;
}

int paddle_beam_decode_with_lm_and_hotwords(at::Tensor th_probs,
                                            at::Tensor th_seq_lens,
                                            void* decoder_options,
                                            void* scorer,
                                            void* hotword_scorer,
                                            at::Tensor th_output,
                                            at::Tensor th_timesteps,
                                            at::Tensor th_scores,
                                            at::Tensor th_out_length)
{

    DecoderOptions* options = static_cast<DecoderOptions*>(decoder_options);

    Scorer* ext_scorer = nullptr;
    if (scorer != nullptr) {
        ext_scorer = static_cast<Scorer*>(scorer);
    }

    HotwordScorer* ext_hotword_scorer = nullptr;
    if (hotword_scorer != nullptr) {
        ext_hotword_scorer = static_cast<HotwordScorer*>(hotword_scorer);
    }

    const int64_t max_time = th_probs.size(1);
    const int64_t batch_size = th_probs.size(0);
    const int64_t num_classes = th_probs.size(2);

    std::vector<std::vector<std::vector<double>>> inputs;
    auto prob_accessor = th_probs.accessor<float, 3>();
    auto seq_len_accessor = th_seq_lens.accessor<int, 1>();

    for (int b = 0; b < batch_size; ++b) {
        // avoid a crash by ensuring that an
        // erroneous seq_len doesn't have us try to access memory
        // we shouldn't
        int seq_len = std::min((int)seq_len_accessor[b], (int)max_time);
        std::vector<std::vector<double>> temp(seq_len, std::vector<double>(num_classes));
        for (int t = 0; t < seq_len; ++t) {
            for (int n = 0; n < num_classes; ++n) {
                float val = prob_accessor[b][t][n];
                temp[t][n] = val;
            }
        }
        inputs.push_back(temp);
    }

    std::vector<std::vector<std::pair<double, Output>>> batch_results
        = ctc_beam_search_decoder_batch(inputs, options, ext_scorer, ext_hotword_scorer);
    auto outputs_accessor = th_output.accessor<int, 3>();
    auto timesteps_accessor = th_timesteps.accessor<int, 3>();
    auto scores_accessor = th_scores.accessor<float, 2>();
    auto out_length_accessor = th_out_length.accessor<int, 2>();

    for (int b = 0; b < batch_results.size(); ++b) {
        std::vector<std::pair<double, Output>> results = batch_results[b];
        for (int p = 0; p < results.size(); ++p) {
            std::pair<double, Output> n_path_result = results[p];
            Output output = n_path_result.second;
            std::vector<int> output_tokens = output.tokens;
            std::vector<int> output_timesteps = output.timesteps;
            for (int t = 0; t < output_tokens.size(); ++t) {
                outputs_accessor[b][p][t] = output_tokens[t]; // fill output tokens
                timesteps_accessor[b][p][t] = output_timesteps[t];
            }
            scores_accessor[b][p] = n_path_result.first;
            out_length_accessor[b][p] = output_tokens.size();
        }
    }
    return 1;
}

int paddle_beam_decode(at::Tensor th_probs,
                       at::Tensor th_seq_lens,
                       void* decoder_options,
                       at::Tensor th_output,
                       at::Tensor th_timesteps,
                       at::Tensor th_scores,
                       at::Tensor th_out_length)
{

    return paddle_beam_decode_with_lm_and_hotwords(th_probs,
                                                   th_seq_lens,
                                                   decoder_options,
                                                   nullptr,
                                                   nullptr,
                                                   th_output,
                                                   th_timesteps,
                                                   th_scores,
                                                   th_out_length);
}

int paddle_beam_decode_with_lm(at::Tensor th_probs,
                               at::Tensor th_seq_lens,
                               void* decoder_options,
                               void* scorer,
                               at::Tensor th_output,
                               at::Tensor th_timesteps,
                               at::Tensor th_scores,
                               at::Tensor th_out_length)
{

    return paddle_beam_decode_with_lm_and_hotwords(th_probs,
                                                   th_seq_lens,
                                                   decoder_options,
                                                   scorer,
                                                   nullptr,
                                                   th_output,
                                                   th_timesteps,
                                                   th_scores,
                                                   th_out_length);
}

int paddle_beam_decode_with_hotwords(at::Tensor th_probs,
                                     at::Tensor th_seq_lens,
                                     void* decoder_options,
                                     void* hotword_scorer,
                                     at::Tensor th_output,
                                     at::Tensor th_timesteps,
                                     at::Tensor th_scores,
                                     at::Tensor th_out_length)
{

    return paddle_beam_decode_with_lm_and_hotwords(th_probs,
                                                   th_seq_lens,
                                                   decoder_options,
                                                   nullptr,
                                                   hotword_scorer,
                                                   th_output,
                                                   th_timesteps,
                                                   th_scores,
                                                   th_out_length);
}

void* paddle_get_decoder_options(std::vector<std::string> vocab,
                                 size_t cutoff_top_n,
                                 double cutoff_prob,
                                 size_t beam_width,
                                 size_t num_processes,
                                 size_t blank_id,
                                 bool log_probs_input,
                                 bool is_bpe_based,
                                 float unk_score,
                                 char token_separator)
{
    DecoderOptions* decoder_options = new DecoderOptions(vocab,
                                                         cutoff_top_n,
                                                         cutoff_prob,
                                                         beam_width,
                                                         num_processes,
                                                         blank_id,
                                                         log_probs_input,
                                                         is_bpe_based,
                                                         unk_score,
                                                         token_separator);
    return static_cast<void*>(decoder_options);
}

void* paddle_get_scorer(double alpha,
                        double beta,
                        const char* lm_path,
                        std::vector<std::string> new_vocab,
                        std::string lm_type,
                        const char* fst_path)
{
    Scorer* scorer = new Scorer(alpha, beta, lm_path, new_vocab, lm_type, fst_path);
    return static_cast<void*>(scorer);
}

void* get_hotword_scorer(void* decoder_options,
                         std::vector<std::vector<std::string>> hotwords,
                         std::vector<float> hotword_weights,
                         char token_separator)
{
    DecoderOptions* options = static_cast<DecoderOptions*>(decoder_options);
    HotwordScorer* scorer = new HotwordScorer(
        options->vocab, hotwords, hotword_weights, token_separator, options->is_bpe_based);
    return static_cast<void*>(scorer);
}

std::pair<torch::Tensor, torch::Tensor>
beam_decode_with_given_state(at::Tensor th_probs,
                             at::Tensor th_seq_lens,
                             size_t num_processes,
                             std::vector<void*>& states,
                             const std::vector<bool>& is_eos_s,
                             at::Tensor th_scores,
                             at::Tensor th_out_length)
{
    const int64_t max_time = th_probs.size(1);
    const int64_t batch_size = th_probs.size(0);
    const int64_t num_classes = th_probs.size(2);

    std::vector<std::vector<std::vector<double>>> inputs;
    auto prob_accessor = th_probs.accessor<float, 3>();
    auto seq_len_accessor = th_seq_lens.accessor<int, 1>();

    for (int b = 0; b < batch_size; ++b) {
        // avoid a crash by ensuring that an erroneous seq_len doesn't have us try to access memory
        // we shouldn't
        int seq_len = std::min((int)seq_len_accessor[b], (int)max_time);
        std::vector<std::vector<double>> temp(seq_len, std::vector<double>(num_classes));
        for (int t = 0; t < seq_len; ++t) {
            for (int n = 0; n < num_classes; ++n) {
                float val = prob_accessor[b][t][n];
                temp[t][n] = val;
            }
        }
        inputs.push_back(temp);
    }

    std::vector<std::vector<std::pair<double, Output>>> batch_results
        = ctc_beam_search_decoder_batch_with_states(inputs, num_processes, states, is_eos_s);

    int max_result_size = 0;
    int max_output_tokens_size = 0;
    for (int b = 0; b < batch_results.size(); ++b) {
        std::vector<std::pair<double, Output>> results = batch_results[b];
        if (batch_results[b].size() > max_result_size) {
            max_result_size = batch_results[b].size();
        }
        for (int p = 0; p < results.size(); ++p) {
            std::pair<double, Output> n_path_result = results[p];
            Output output = n_path_result.second;
            std::vector<int> output_tokens = output.tokens;

            if (output_tokens.size() > max_output_tokens_size) {
                max_output_tokens_size = output_tokens.size();
            }
        }
    }

    torch::Tensor output_tokens_tensor
        = torch::randint(1, { batch_results.size(), max_result_size, max_output_tokens_size });
    torch::Tensor output_timesteps_tensor
        = torch::randint(1, { batch_results.size(), max_result_size, max_output_tokens_size });

    auto scores_accessor = th_scores.accessor<float, 2>();
    auto out_length_accessor = th_out_length.accessor<int, 2>();

    for (int b = 0; b < batch_results.size(); ++b) {
        std::vector<std::pair<double, Output>> results = batch_results[b];
        for (int p = 0; p < results.size(); ++p) {
            std::pair<double, Output> n_path_result = results[p];
            Output output = n_path_result.second;
            std::vector<int> output_tokens = output.tokens;
            std::vector<int> output_timesteps = output.timesteps;
            for (int t = 0; t < output_tokens.size(); ++t) {
                output_tokens_tensor[b][p][t] = output_tokens[t]; // fill output tokens
                output_timesteps_tensor[b][p][t] = output_timesteps[t];
            }
            scores_accessor[b][p] = n_path_result.first;
            out_length_accessor[b][p] = output_tokens.size();
        }
    }

    return { output_tokens_tensor, output_timesteps_tensor };
}

std::pair<torch::Tensor, torch::Tensor>
paddle_beam_decode_with_given_state(at::Tensor th_probs,
                                    at::Tensor th_seq_lens,
                                    size_t num_processes,
                                    std::vector<void*> states,
                                    std::vector<bool> is_eos_s,
                                    at::Tensor th_scores,
                                    at::Tensor th_out_length)
{

    return beam_decode_with_given_state(
        th_probs, th_seq_lens, num_processes, states, is_eos_s, th_scores, th_out_length);
}

void* paddle_get_decoder_state(void* decoder_options, void* scorer)
{
    // DecoderState state(vocabulary, beam_size, cutoff_prob, cutoff_top_n, blank_id, log_input,
    // ext_scorer);
    DecoderOptions* options = static_cast<DecoderOptions*>(decoder_options);
    Scorer* ext_scorer = nullptr;
    if (scorer != nullptr) {
        ext_scorer = static_cast<Scorer*>(scorer);
    }
    DecoderState* state = new DecoderState(options, ext_scorer, nullptr);
    return static_cast<void*>(state);
}

void paddle_release_state(void* state) { delete static_cast<DecoderState*>(state); }

void paddle_release_scorer(void* scorer) { delete static_cast<Scorer*>(scorer); }

void paddle_release_decoder_options(void* decoder_options)
{
    delete static_cast<DecoderOptions*>(decoder_options);
}

void paddle_release_hotword_scorer(void* scorer)
{
    delete static_cast<HotwordScorer*>(scorer);
    scorer = nullptr;
}

int is_character_based(void* scorer)
{
    Scorer* ext_scorer = static_cast<Scorer*>(scorer);
    return ext_scorer->is_character_based();
}

size_t get_max_order(void* scorer)
{
    Scorer* ext_scorer = static_cast<Scorer*>(scorer);
    return ext_scorer->get_max_order();
}

size_t get_lexicon_size(void* scorer)
{
    Scorer* ext_scorer = static_cast<Scorer*>(scorer);
    return ext_scorer->get_lexicon_size();
}

void reset_params(void* scorer, double alpha, double beta)
{
    Scorer* ext_scorer = static_cast<Scorer*>(scorer);
    ext_scorer->reset_params(alpha, beta);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("paddle_beam_decode", &paddle_beam_decode, "paddle_beam_decode");
    m.def("paddle_beam_decode_with_lm", &paddle_beam_decode_with_lm, "paddle_beam_decode_with_lm");
    m.def("paddle_beam_decode_with_hotwords",
          &paddle_beam_decode_with_hotwords,
          "paddle_beam_decode_with_hotwords");
    m.def("paddle_beam_decode_with_lm_and_hotwords",
          &paddle_beam_decode_with_lm_and_hotwords,
          "paddle_beam_decode_with_lm_and_hotwords",
          py::arg("th_probs"),
          py::arg("th_seq_lens"),
          py::arg("decoder_options"),
          py::arg("scorer").none(true),
          py::arg("hotword_scorer").none(true),
          py::arg("th_output"),
          py::arg("th_timestamps"),
          py::arg("th_scores"),
          py::arg("th_out_length"));
    m.def("paddle_get_decoder_options", &paddle_get_decoder_options, "paddle_get_decoder_options");
    m.def("paddle_get_scorer", &paddle_get_scorer, "paddle_get_scorer");
    m.def("get_hotword_scorer", &get_hotword_scorer, "get_hotword_scorer");
    m.def("paddle_release_scorer", &paddle_release_scorer, "paddle_release_scorer");
    m.def("paddle_release_decoder_options",
          &paddle_release_decoder_options,
          "paddle_release_decoder_options");
    m.def("paddle_release_hotword_scorer",
          &paddle_release_hotword_scorer,
          "paddle_release_hotword_scorer");
    m.def("is_character_based", &is_character_based, "is_character_based");
    m.def("get_max_order", &get_max_order, "get_max_order");
    m.def("get_lexicon_size", &get_lexicon_size, "get_max_order");
    m.def("reset_params", &reset_params, "reset_params");
    m.def("paddle_get_decoder_state", &paddle_get_decoder_state, "paddle_get_decoder_state");
    m.def("paddle_beam_decode_with_given_state",
          &paddle_beam_decode_with_given_state,
          "paddle_beam_decode_with_given_state");
    m.def("paddle_release_state", &paddle_release_state, "paddle_release_state");
    // paddle_beam_decode_with_given_state
}
