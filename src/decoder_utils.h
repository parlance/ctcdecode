#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "output.h"
#include "path_trie.h"

namespace ctcdecode {

const float NUM_FLT_INF = std::numeric_limits<float>::max();
const float NUM_FLT_MIN = std::numeric_limits<float>::min();
const float NUM_FLT_LOGE = 0.4342944819;

// Function template for comparing two pairs
template <typename T1, typename T2>
bool pair_comp_second_rev(const std::pair<T1, T2> &a,
                          const std::pair<T1, T2> &b) {
  return a.second > b.second;
}

// Return the sum of two probabilities in log scale
template <typename T> T log_sum_exp(const T &x, const T &y) {
  static T num_min = -std::numeric_limits<T>::max();
  if (x <= num_min)
    return y;
  if (y <= num_min)
    return x;
  T xmax = std::max(x, y);
  return std::log(std::exp(x - xmax) + std::exp(y - xmax)) + xmax;
}

// Get pruned probability vector for each time step's beam search
std::vector<std::pair<size_t, float>>
get_pruned_log_probs(const std::vector<double> &prob_step, double cutoff_prob,
                     size_t cutoff_top_n, int log_input);

// Get beam search result from prefixes in trie tree
std::vector<std::pair<double, Output>>
get_beam_search_result(const std::vector<PathTrie *> &prefixes,
                       size_t beam_size);

// Functor for prefix comparison
bool prefix_compare(const PathTrie *x, const PathTrie *y);

bool prefix_compare_external_scores(
    const PathTrie *x, const PathTrie *y,
    const std::unordered_map<const PathTrie *, float> &scores);

} // namespace ctcdecode
