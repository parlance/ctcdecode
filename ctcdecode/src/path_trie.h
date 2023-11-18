#ifndef PATH_TRIE_H
#define PATH_TRIE_H

#include <algorithm>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "fst/fstlib.h"

/* Trie tree for prefix storing and manipulating, with a dictionary in
 * finite-state transducer for spelling correction.
 */
class PathTrie {
public:
    PathTrie();
    ~PathTrie();

    // get new prefix after appending new char
    PathTrie* get_path_trie(int new_char,
                            int new_timestep,
                            float log_prob_c,
                            bool reset = true,
                            bool check_lexicon = true);

    // get the prefix in index from root to current node
    PathTrie* get_path_vec(std::vector<int>& output, std::vector<int>& timesteps);

    // get the prefix in index from some stop node to current nodel
    PathTrie* get_path_vec(std::vector<int>& output,
                           std::vector<int>& timesteps,
                           int stop,
                           size_t max_steps = std::numeric_limits<size_t>::max());

    // creates new PathTrie* node
    PathTrie* create_new_node(int new_char, int new_timestep, float cur_log_prob_c);

    // update log probs
    void iterate_to_vec(std::vector<PathTrie*>& output);

    // set lexicon for FST
    void set_lexicon(fst::StdVectorFst* lexicon);

    void set_matcher(std::shared_ptr<fst::SortedMatcher<fst::StdVectorFst>>);

    bool is_empty() { return ROOT_ == character; }

    bool is_hotpath() { return is_hotpath_; }

    // set as hotpath
    void mark_as_hotpath() { is_hotpath_ = true; }

    bool is_word_start_char() { return is_word_start_char_; }

    // set as word start character
    void mark_as_word_start_char() { is_word_start_char_ = true; }

    bool has_lexicon() { return has_lexicon_; }

    // check if current token forms OOV word
    bool is_oov_token();

    // remove current path from root
    void remove();

    void reset_hotword_params();
    void copy_parent_hotword_params();

    float log_prob_b_prev;
    float log_prob_nb_prev;
    float log_prob_b_cur;
    float log_prob_nb_cur;
    float log_prob_b_prev_hw;
    float log_prob_nb_prev_hw;
    float log_prob_b_cur_hw;
    float log_prob_nb_cur_hw;

    float log_prob_c;
    float score;
    float score_hw;
    float approx_ctc;
    int character;
    int timestep;
    PathTrie* parent;
    float hotword_score;
    int shortest_unigram_length;
    float hotword_weight;
    std::string partial_hotword;
    std::shared_ptr<fst::SortedMatcher<fst::StdVectorFst>> hotword_matcher;
    fst::StdVectorFst::StateId hotword_dictionary_state;
    int hotword_match_len;

private:
    int ROOT_;
    bool exists_;
    bool has_lexicon_;

    bool is_hotpath_;
    bool is_word_start_char_;

    std::vector<std::pair<int, PathTrie*>> children_;

    // pointer to lexicon of FST
    fst::StdVectorFst* lexicon_;
    fst::StdVectorFst::StateId lexicon_state_;
    // true if finding ars in FST
    std::shared_ptr<fst::SortedMatcher<fst::StdVectorFst>> matcher_;
};

#endif // PATH_TRIE_H
