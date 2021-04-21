#include "decoder_utils.h"

namespace ctcdecode {

std::vector<std::pair<size_t, float>>
get_pruned_log_probs(const std::vector<double> &prob_step, double cutoff_prob,
                     size_t cutoff_top_n, int log_input) {
  std::vector<std::pair<int, double>> prob_idx;
  double log_cutoff_prob = log(cutoff_prob);
  for (size_t i = 0; i < prob_step.size(); ++i) {
    prob_idx.push_back(std::pair<int, double>(i, prob_step[i]));
  }
  // pruning of vacobulary
  size_t cutoff_len = prob_step.size();
  if (log_cutoff_prob < 0.0 || cutoff_top_n < cutoff_len) {
    std::sort(prob_idx.begin(), prob_idx.end(),
              pair_comp_second_rev<int, double>);
    if (log_cutoff_prob < 0.0) {
      double cum_prob = 0.0;
      cutoff_len = 0;
      for (size_t i = 0; i < prob_idx.size(); ++i) {
        cum_prob = log_sum_exp(cum_prob, log_input ? prob_idx[i].second
                                                   : log(prob_idx[i].second));
        cutoff_len += 1;
        if (cum_prob >= cutoff_prob || cutoff_len >= cutoff_top_n)
          break;
      }
    } else {
      cutoff_len = cutoff_top_n;
    }
    prob_idx = std::vector<std::pair<int, double>>(
        prob_idx.begin(), prob_idx.begin() + cutoff_len);
  }
  std::vector<std::pair<size_t, float>> log_prob_idx;
  for (size_t i = 0; i < cutoff_len; ++i) {
    log_prob_idx.push_back(std::pair<int, float>(
        prob_idx[i].first, log_input ? prob_idx[i].second
                                     : log(prob_idx[i].second + NUM_FLT_MIN)));
  }
  return log_prob_idx;
}

std::vector<std::pair<double, Output>>
get_beam_search_result(const std::vector<PathTrie *> &prefixes,
                       size_t beam_size) {
  // allow for the post processing
  std::vector<PathTrie *> space_prefixes;
  if (space_prefixes.empty()) {
    for (size_t i = 0; i < beam_size && i < prefixes.size(); ++i) {
      space_prefixes.push_back(prefixes[i]);
    }
  }

  std::sort(space_prefixes.begin(), space_prefixes.end(), prefix_compare);
  std::vector<std::pair<double, Output>> output_vecs;
  for (size_t i = 0; i < beam_size && i < space_prefixes.size(); ++i) {
    std::vector<int> output;
    std::vector<int> timesteps;
    space_prefixes[i]->get_path_vec(output, timesteps);
    Output outputs;
    outputs.tokens = output;
    outputs.timesteps = timesteps;
    std::pair<double, Output> output_pair(-space_prefixes[i]->approx_ctc,
                                          outputs);
    output_vecs.emplace_back(output_pair);
  }

  return output_vecs;
}

bool prefix_compare(const PathTrie *x, const PathTrie *y) {
  if (x->score == y->score) {
    if (x->character == y->character) {
      return false;
    } else {
      return (x->character < y->character);
    }
  } else {
    return x->score > y->score;
  }
}

bool prefix_compare_external_scores(
    const PathTrie *x, const PathTrie *y,
    const std::unordered_map<const PathTrie *, float> &scores) {
  if (scores.at(x) == scores.at(y)) {
    if (x->character == y->character) {
      return false;
    } else {
      return (x->character < y->character);
    }
  } else {
    return scores.at(x) > scores.at(y);
  }
}
} // namespace ctcdecode
