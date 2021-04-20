#include <vector>
#include <utility>

#include <torch/script.h>
#include <ATen/Parallel.h>

#include "ctc_beam_search_decoder.h"

namespace ctcdecode {
namespace {

/* CTC Beam Search Decoder for batch data

 * Parameters:
 *     probs_seq: 3-D vector that each element is a 2-D vector that can be used
 *                by ctc_beam_search_decoder().
 *     vocabulary: A vector of vocabulary.
 *     beam_size: The width of beam search.
 *     num_processes: Number of threads for beam search.
 *     cutoff_prob: Cutoff probability for pruning.
 *     cutoff_top_n: Cutoff number for pruning.
 * Return:
 *     A 2-D vector that each element is a vector of beam search decoding
 *     result for one audio sample.
*/
std::vector<std::vector<std::pair<double, Output>>>
ctc_beam_search_decoder_batch(
    const std::vector<std::vector<std::vector<double>>> &probs_split,
    const std::vector<std::string> &vocabulary,
    size_t beam_size,
    size_t num_processes,
    double cutoff_prob,
    size_t cutoff_top_n,
    size_t blank_id,
    int log_input)
{
  size_t batch_size = probs_split.size();
  auto grain_size = batch_size / num_processes;

  std::vector<std::vector<std::pair<double, Output>>> results(batch_size);

  at::parallel_for(0, batch_size, grain_size, [&](int64_t begin, int64_t end) {
    for (auto i = begin; i < end; ++i) {
      results[i] = ctc_beam_search_decoder(probs_split[i],
                                          vocabulary,
                                          beam_size,
                                          cutoff_prob,
                                          cutoff_top_n,
                                          blank_id,
                                          log_input);
    }
  });
  return results;
}

using decoder_output = std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>;


decoder_output beam_decode(torch::Tensor th_probs,
                torch::Tensor th_seq_lens,
                std::vector<std::string> new_vocab,
                int64_t vocab_size,
                int64_t beam_size,
                int64_t num_processes,
                double cutoff_prob,
                int64_t cutoff_top_n,
                int64_t blank_id,
                bool log_input)
{
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
    ctc_beam_search_decoder_batch(
        inputs, new_vocab, beam_size, num_processes, cutoff_prob, cutoff_top_n, blank_id, log_input);

    std::cout << "foo\n" << std::endl << std::flush;
    auto max_seq_len = th_probs.size(1);
    auto beams = torch::empty({batch_size, beam_size, max_seq_len}, torch::kInt32);
    auto lengths = torch::zeros({batch_size, beam_size}, torch::kInt32);
    auto scores = torch::empty({batch_size, beam_size}, torch::kFloat);
    auto timesteps = torch::empty({batch_size, beam_size, max_seq_len}, torch::kInt32);

    auto outputs_accessor = beams.accessor<int, 3>();
    auto out_length_accessor =  lengths.accessor<int, 2>();
    auto scores_accessor =  scores.accessor<float, 2>();
    auto timesteps_accessor =  timesteps.accessor<int, 3>();

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
    return std::make_tuple(beams, lengths, scores, timesteps);
}

TORCH_LIBRARY(ctcdecode, m) {
  m.def("beam_decode", &beam_decode);
}

} // namespace
} // namespace ctcdecode
