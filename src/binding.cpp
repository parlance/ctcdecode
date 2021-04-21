#include <utility>
#include <vector>

#include <ATen/Parallel.h>
#include <torch/script.h>

#include "ctc_beam_search_decoder.h"

namespace ctcdecode {
namespace {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
beam_decode(torch::Tensor probs, c10::optional<torch::Tensor> seq_lens_,
            std::vector<std::string> vocabulary, int64_t beam_size,
            int64_t cutoff_top_n, c10::optional<double> cutoff_prob_,
            int64_t blank_id, bool is_nll, int64_t num_processes) {

  const double cutoff_prob = cutoff_prob_.value_or(1.1);
  const int64_t num_classes = vocabulary.size();

  TORCH_CHECK(probs.ndimension() == 3, "`probs` has to be 3D Tensor.");
  TORCH_CHECK(probs.device().is_cpu(), "`probs` has to be on CPU.");
  TORCH_CHECK(probs.dtype() == torch::kFloat,
              "`probs` has to be float Tensor.");
  TORCH_CHECK(
      probs.size(2) == num_classes,
      "The 3rd dimension of `probs` has to match the size of the vocabulary.");

  const int64_t batch_size = probs.size(0);
  const int64_t max_seq_len = probs.size(1);

  auto seq_lens = [&]() {
    if (seq_lens_.has_value()) {
      auto seq_lens = seq_lens_.value();
      TORCH_CHECK(seq_lens.ndimension() == 1,
                  "When provided, `seq_lens` has to be 1D Tensor.");
      TORCH_CHECK(seq_lens.size(0) == batch_size,
                  "When provided, `seq_lens` has to have the same batch size "
                  "as `probs`.");
      TORCH_CHECK(seq_lens.device().is_cpu(),
                  "When provided, `seq_lens` has to be on CPU.");
      TORCH_CHECK(seq_lens.dtype() == torch::kInt32,
                  "When provided, `seq_lens` has to be on CPU.");
      TORCH_CHECK((seq_lens <= max_seq_len).all().item<bool>(),
                  "All the values in`seq_lens` must be less than or equal to "
                  "the length of `probs`.");
      return seq_lens;
    }
    return torch::full({batch_size}, max_seq_len, torch::kInt32);
  }();

  // Copy data into the format ctc beam search expects
  std::vector<std::vector<std::vector<double>>> inputs;
  auto prob_accessor = probs.accessor<float, 3>();
  auto seq_len_accessor = seq_lens.accessor<int, 1>();

  // TODO: use slicing and memcpy OR pass Tensor directly to ctc_beam_search_decoder
  for (int64_t b = 0; b < batch_size; ++b) {
    int seq_len = (int)seq_len_accessor[b];
    std::vector<std::vector<double>> temp(seq_len,
                                          std::vector<double>(num_classes));
    for (int t = 0; t < seq_len; ++t) {
      for (int n = 0; n < num_classes; ++n) {
        float val = prob_accessor[b][t][n];
        temp[t][n] = val;
      }
    }
    inputs.push_back(temp);
  }

  std::vector<std::vector<std::pair<double, Output>>> batch_results(batch_size);
  auto grain_size = batch_size / num_processes;
  at::parallel_for(0, batch_size, grain_size, [&](int64_t begin, int64_t end) {
    for (auto i = begin; i < end; ++i) {
      batch_results[i] =
          ctc_beam_search_decoder(inputs[i], vocabulary, beam_size, cutoff_prob,
                                  cutoff_top_n, blank_id, is_nll);
    }
  });

  auto beams =
      torch::empty({batch_size, beam_size, max_seq_len}, torch::kInt32);
  auto output_lengths = torch::zeros({batch_size, beam_size}, torch::kInt32);
  auto scores = torch::empty({batch_size, beam_size}, torch::kFloat);
  auto timesteps =
      torch::empty({batch_size, beam_size, max_seq_len}, torch::kInt32);

  auto outputs_accessor = beams.accessor<int, 3>();
  auto output_lengths_accessor = output_lengths.accessor<int, 2>();
  auto scores_accessor = scores.accessor<float, 2>();
  auto timesteps_accessor = timesteps.accessor<int, 3>();

  // TODO: use slice and memcpy
  for (size_t b = 0; b < batch_results.size(); ++b) {
    std::vector<std::pair<double, Output>> results = batch_results[b];
    for (size_t p = 0; p < results.size(); ++p) {
      std::pair<double, Output> n_path_result = results[p];
      Output output = n_path_result.second;
      std::vector<int> output_tokens = output.tokens;
      std::vector<int> output_timesteps = output.timesteps;
      for (size_t t = 0; t < output_tokens.size(); ++t) {
        outputs_accessor[b][p][t] = output_tokens[t]; // fill output tokens
        timesteps_accessor[b][p][t] = output_timesteps[t];
      }
      scores_accessor[b][p] = n_path_result.first;
      output_lengths_accessor[b][p] = output_tokens.size();
    }
  }
  return std::make_tuple(beams, output_lengths, scores, timesteps);
}

TORCH_LIBRARY(ctcdecode, m) { m.def("beam_decode", &beam_decode); }

} // namespace
} // namespace ctcdecode
