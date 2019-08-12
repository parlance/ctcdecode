#ifndef DECODERSTATE_H_
#define DECODERSTATE_H_

#include <vector>
#include <string>

class PathTrie;
class Scorer;

/* Streaming state of the decoder, containing the prefixes and initial root prefix plus state variables. */

struct DecoderState {
  int abs_time_step;
  int space_id;
  size_t beam_size;
  double cutoff_prob;
  size_t cutoff_top_n;
  size_t blank_id;
  int log_input;
  std::vector<std::string> vocabulary;
  Scorer *ext_scorer;

  std::vector<PathTrie*> prefixes;
  PathTrie *prefix_root;

  ~DecoderState() {
    if (prefix_root != nullptr) {
      delete prefix_root;
    }
    prefix_root = nullptr;
  }
};

#endif  // DECODERSTATE_H_
