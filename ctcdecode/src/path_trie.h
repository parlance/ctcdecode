#ifndef PATH_TRIE_H
#define PATH_TRIE_H

#include <algorithm>
#include <limits>
#include <memory>
#include <utility>
#include <vector>
#include <map>

#include "fst/fstlib.h"

/* Trie tree for prefix storing and manipulating, with a dictionary in
 * finite-state transducer for spelling correction.
 */
class PathTrie {
public:
  PathTrie();
  ~PathTrie();

  // get new prefix after appending new char
  PathTrie* get_path_trie(int new_char, int new_timestep, float log_prob_c, bool reset = true);

  // get the prefix in index from root to current node
  PathTrie* get_path_vec(std::vector<int>& output, std::vector<int>& timesteps);

  // get the prefix in index from some stop node to current nodel
  PathTrie* get_path_vec(std::vector<int>& output,
                         std::vector<int>& timesteps,
                         int stop,
                         size_t max_steps = std::numeric_limits<size_t>::max());

  // update log probs
  void iterate_to_vec(std::vector<PathTrie*>& output);

  // set dictionary for FST
  void set_dictionary(fst::StdVectorFst* dictionary);

  void set_matcher(std::shared_ptr<fst::SortedMatcher<fst::StdVectorFst>>);

  bool is_empty() { return ROOT_ == character; }

  // remove current path from root
  void remove();

  /// funnels

  // create and set mini dictionary
  double create_mini_dictionary(const std::map<std::string, std::string> &funnels,
                                std::unordered_map<std::string, int> char_map_,
                                int space_id,
                                bool add_space);

  // // prepare matcher for mini dictionary
  // void set_mini_matcher(std::shared_ptr<fst::SortedMatcher<fst::StdVectorFst>>);


  float log_prob_b_prev;
  float log_prob_nb_prev;
  float log_prob_b_cur;
  float log_prob_nb_cur;
  float log_prob_c;
  float score;
  float approx_ctc;
  int character;
  int timestep;
  PathTrie* parent;
  std::vector<std::string> vocab_tmp;      // 文字列配列

private:
  int ROOT_;
  bool exists_;
  bool has_dictionary_;

  std::vector<std::pair<int, PathTrie*>> children_;

  // pointer to dictionary of FST
  fst::StdVectorFst* dictionary_;
  fst::StdVectorFst::StateId dictionary_state_;
  // true if finding ars in FST
  std::shared_ptr<fst::SortedMatcher<fst::StdVectorFst>> matcher_;

  /// funnels

  // flag if mini dictionary exists
  bool has_mini_dictionary_;

  // pointer to mnini dictionary of FST
  fst::StdVectorFst* mini_dictionary_;

  // miin dictionary state
  fst::StdVectorFst::StateId mini_dictionary_state_;

  // true if finding ars in mini dictionary FST
  std::shared_ptr<fst::SortedMatcher<fst::StdVectorFst>> mini_matcher_;
};

#endif  // PATH_TRIE_H
