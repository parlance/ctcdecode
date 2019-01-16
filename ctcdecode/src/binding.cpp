#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <torch/torch.h>
#include "scorer.h"
#include "ctc_beam_search_decoder.h"
#include "utf8.h"

int utf8_to_utf8_char_vec(const char* labels, std::vector<std::string>& new_vocab) {
    const char* str_i = labels;
    const char* end = str_i + strlen(labels)+1;
    do {
        char u[5] = {0,0,0,0,0};
        uint32_t code = utf8::next(str_i, end);
        if (code == 0) {
            continue;
        }
        utf8::append(code, u);
        new_vocab.push_back(std::string(u));
    }
    while (str_i < end);
}

int beam_decode(at::Tensor th_probs,
                at::Tensor th_seq_lens,
                const char* labels,
                int vocab_size,
                size_t beam_size,
                size_t num_processes,
                double cutoff_prob,
                size_t cutoff_top_n,
                size_t blank_id,
                bool log_input,
                void *scorer,
                at::Tensor th_output,
                at::Tensor th_timesteps,
                at::Tensor th_scores,
                at::Tensor th_out_length)
{
    std::vector<std::string> new_vocab;
    utf8_to_utf8_char_vec(labels, new_vocab);
    Scorer *ext_scorer = NULL;
    if (scorer != NULL) {
        ext_scorer = static_cast<Scorer *>(scorer);
    }
    const int64_t max_time = th_probs.size(1);
    const int64_t batch_size = th_probs.size(0);
    const int64_t num_classes = th_probs.size(2);

    std::vector<std::vector<std::vector<double>>> inputs;
    auto prob_accessor = th_probs.accessor<float, 3>();
    auto seq_len_accessor = th_seq_lens.accessor<int, 1>();

    for (int b=0; b < batch_size; ++b) {
        // avoid a crash by ensuring that an erroneous seq_len doesn't have us try to access memory we shouldn't
        int seq_len = std::min((int)seq_len_accessor[b], (int)max_time);
        std::vector<std::vector<double>> temp (seq_len, std::vector<double>(num_classes));
        for (int t=0; t < seq_len; ++t) {
            for (int n=0; n < num_classes; ++n) {
                float val = prob_accessor[b][t][n];
                temp[t][n] = val;
            }
        }
        inputs.push_back(temp);
    }

    std::vector<std::vector<std::pair<double, Output>>> batch_results =
    ctc_beam_search_decoder_batch(inputs, new_vocab, beam_size, num_processes, cutoff_prob, cutoff_top_n, blank_id, log_input, ext_scorer);
    auto outputs_accessor =  th_output.accessor<int, 3>();
    auto timesteps_accessor =  th_timesteps.accessor<int, 3>();
    auto scores_accessor =  th_scores.accessor<float, 2>();
    auto out_length_accessor =  th_out_length.accessor<int, 2>();


    for (int b = 0; b < batch_results.size(); ++b){
        std::vector<std::pair<double, Output>> results = batch_results[b];
        for (int p = 0; p < results.size();++p){
            std::pair<double, Output> n_path_result = results[p];
            Output output = n_path_result.second;
            std::vector<int> output_tokens = output.tokens;
            std::vector<int> output_timesteps = output.timesteps;
            for (int t = 0; t < output_tokens.size(); ++t){
                outputs_accessor[b][p][t] =  output_tokens[t]; // fill output tokens
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
                       const char* labels,
                       int vocab_size,
                       size_t beam_size,
                       size_t num_processes,
                       double cutoff_prob,
                       size_t cutoff_top_n,
                       size_t blank_id,
                       int log_input,
                       at::Tensor th_output,
                       at::Tensor th_timesteps,
                       at::Tensor th_scores,
                       at::Tensor th_out_length){

    return beam_decode(th_probs, th_seq_lens, labels, vocab_size, beam_size, num_processes,
                cutoff_prob, cutoff_top_n, blank_id, log_input, NULL, th_output, th_timesteps, th_scores, th_out_length);
}

int paddle_beam_decode_lm(at::Tensor th_probs,
                          at::Tensor th_seq_lens,
                          const char* labels,
                          int vocab_size,
                          size_t beam_size,
                          size_t num_processes,
                          double cutoff_prob,
                          size_t cutoff_top_n,
                          size_t blank_id,
                          int log_input,
                          void *scorer,
                          at::Tensor th_output,
                          at::Tensor th_timesteps,
                          at::Tensor th_scores,
                          at::Tensor th_out_length){

    return beam_decode(th_probs, th_seq_lens, labels, vocab_size, beam_size, num_processes,
                cutoff_prob, cutoff_top_n, blank_id, log_input, scorer, th_output, th_timesteps, th_scores, th_out_length);
}


void* paddle_get_scorer(double alpha,
                        double beta,
                        const char* lm_path,
                        const char* labels,
                        int vocab_size) {
    std::vector<std::string> new_vocab;
    utf8_to_utf8_char_vec(labels, new_vocab);
    Scorer* scorer = new Scorer(alpha, beta, lm_path, new_vocab);
    return static_cast<void*>(scorer);
}

int is_character_based(void *scorer){
    Scorer *ext_scorer  = static_cast<Scorer *>(scorer);
    return ext_scorer->is_character_based();
}
size_t get_max_order(void *scorer){
    Scorer *ext_scorer  = static_cast<Scorer *>(scorer);
    return ext_scorer->get_max_order();
}
size_t get_dict_size(void *scorer){
    Scorer *ext_scorer  = static_cast<Scorer *>(scorer);
    return ext_scorer->get_dict_size();
}

void reset_params(void *scorer, double alpha, double beta){
    Scorer *ext_scorer  = static_cast<Scorer *>(scorer);
    ext_scorer->reset_params(alpha, beta);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("paddle_beam_decode", &paddle_beam_decode, "paddle_beam_decode");
  m.def("paddle_beam_decode_lm", &paddle_beam_decode_lm, "paddle_beam_decode_lm");
  m.def("paddle_get_scorer", &paddle_get_scorer, "paddle_get_scorer");
  m.def("is_character_based", &is_character_based, "is_character_based");
  m.def("get_max_order", &get_max_order, "get_max_order");
  m.def("get_dict_size", &get_dict_size, "get_max_order");
  m.def("reset_params", &reset_params, "reset_params");
}
