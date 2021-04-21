#include "ctc_beam_search_decoder.h"
#include "decoder_utils.h"
#include "path_trie.h"

namespace ctcdecode {

DecoderState::DecoderState(const std::vector<std::string> &vocabulary,
                           size_t beam_size, double cutoff_prob,
                           size_t cutoff_top_n, size_t blank_id, int log_input)
    : abs_time_step(0), beam_size(beam_size), cutoff_prob(cutoff_prob),
      cutoff_top_n(cutoff_top_n), blank_id(blank_id), log_input(log_input),
      vocabulary(vocabulary) {
  // assign space id
  auto it = std::find(vocabulary.begin(), vocabulary.end(), " ");
  // if no space in vocabulary
  if (it == vocabulary.end()) {
    space_id = -2;
  } else {
    space_id = std::distance(vocabulary.begin(), it);
  }

  // init prefixes' root
  root.score = root.log_prob_b_prev = 0.0;
  prefixes.push_back(&root);
}

void DecoderState::next(const std::vector<std::vector<double>> &probs_seq) {
  // prefix search over time
  size_t num_time_steps = probs_seq.size();
  for (size_t time_step = 0; time_step < num_time_steps;
       ++time_step, ++abs_time_step) {
    auto &prob = probs_seq[time_step];

    float min_cutoff = -NUM_FLT_INF;
    bool full_beam = false;

    std::vector<std::pair<size_t, float>> log_prob_idx =
        get_pruned_log_probs(prob, cutoff_prob, cutoff_top_n, log_input);
    // loop over chars
    for (size_t index = 0; index < log_prob_idx.size(); index++) {
      auto c = log_prob_idx[index].first;
      auto log_prob_c = log_prob_idx[index].second;

      for (size_t i = 0; i < prefixes.size() && i < beam_size; ++i) {
        auto prefix = prefixes[i];
        if (full_beam && log_prob_c + prefix->score < min_cutoff) {
          break;
        }
        // blank
        if (c == blank_id) {
          prefix->log_prob_b_cur =
              log_sum_exp(prefix->log_prob_b_cur, log_prob_c + prefix->score);
          continue;
        }
        // repeated character
        if (static_cast<int>(c) == prefix->character) {
          prefix->log_prob_nb_cur = log_sum_exp(
              prefix->log_prob_nb_cur, log_prob_c + prefix->log_prob_nb_prev);
        }
        // get new prefix
        auto prefix_new = prefix->get_path_trie(c, abs_time_step, log_prob_c);

        if (prefix_new != nullptr) {
          float log_p = -NUM_FLT_INF;

          if (static_cast<int>(c) == prefix->character &&
              prefix->log_prob_b_prev > -NUM_FLT_INF) {
            log_p = log_prob_c + prefix->log_prob_b_prev;
          } else if (static_cast<int>(c) != prefix->character) {
            log_p = log_prob_c + prefix->score;
          }

          prefix_new->log_prob_nb_cur =
              log_sum_exp(prefix_new->log_prob_nb_cur, log_p);
        }
      } // end of loop over prefix
    }   // end of loop over vocabulary

    prefixes.clear();
    // update log probs
    root.iterate_to_vec(prefixes);

    // only preserve top beam_size prefixes
    if (prefixes.size() >= beam_size) {
      std::nth_element(prefixes.begin(), prefixes.begin() + beam_size,
                       prefixes.end(), prefix_compare);
      for (size_t i = beam_size; i < prefixes.size(); ++i) {
        prefixes[i]->remove();
      }

      prefixes.resize(beam_size);
    }
  } // end of loop over time
}

std::vector<std::pair<double, Output>> DecoderState::decode() const {
  std::vector<PathTrie *> prefixes_copy = prefixes;
  std::unordered_map<const PathTrie *, float> scores;
  for (PathTrie *prefix : prefixes_copy) {
    scores[prefix] = prefix->score;
  }

  using namespace std::placeholders;
  size_t num_prefixes = std::min(prefixes_copy.size(), beam_size);
  std::sort(prefixes_copy.begin(), prefixes_copy.begin() + num_prefixes,
            std::bind(prefix_compare_external_scores, _1, _2, scores));

  // compute aproximate ctc score as the return score, without affecting the
  // return order of decoding result. To delete when decoder gets stable.
  for (size_t i = 0; i < beam_size && i < prefixes_copy.size(); ++i) {
    double approx_ctc = scores[prefixes_copy[i]];
    prefixes_copy[i]->approx_ctc = approx_ctc;
  }

  return get_beam_search_result(prefixes_copy, beam_size);
}

std::vector<std::pair<double, Output>>
ctc_beam_search_decoder(const std::vector<std::vector<double>> &probs_seq,
                        const std::vector<std::string> &vocabulary,
                        size_t beam_size, double cutoff_prob,
                        size_t cutoff_top_n, size_t blank_id, int log_input) {
  DecoderState state(vocabulary, beam_size, cutoff_prob, cutoff_top_n, blank_id,
                     log_input);
  state.next(probs_seq);
  return state.decode();
}

} // namespace ctcdecode
