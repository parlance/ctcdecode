#ifndef HOTWORD_SCORER_H_
#define HOTWORD_SCORER_H_

#include <unordered_map>

#include "fst/fstlib.h"
#include "path_trie.h"

class HotwordScorer {
public:
    /* Initialize HotwordScorer for CTC beam decoding
     *
     * Parameters:
     *     vocab_list: A vector of vocabulary (labels).
     *     hotwords: A vector of hotwords containing character/token vectors
     *     hotword_weights: A vector of weights corresponds to each hotword in `hotwords`.
     *     token_separator: Token seperator character for bpe based vocabulary.
     *     is_bpe_based: Whether the vocabulary is bpe based
     */
    HotwordScorer(const std::vector<std::string>& vocab_list,
                  const std::vector<std::vector<std::string>>& hotwords,
                  const std::vector<float>& hotword_weights,
                  char token_separator,
                  bool is_bpe_based);
    ~HotwordScorer();

    size_t get_hotword_dict_size() const { return dict_size_; }

    bool is_bpe_based() const { return is_bpe_based_; }

    bool is_hotpath(PathTrie* path, int space_id, int apostrophe_id);
    void estimate_hw_score(PathTrie* path);
    std::tuple<std::string, int> find_shortest_candidate_hotword_length(PathTrie*);
    bool is_char_extendable_from_state(PathTrie* path, fst::StdVectorFst::StateId dict_state);

    fst::StdVectorFst* dictionary;
    std::vector<float> hotword_weights;
    std::vector<std::vector<std::string>> hotwords;
    fst::TropicalWeight FSTZERO;
    char token_separator;

protected:
    void setup(const std::vector<std::string>& vocab_list);

    // fill hotword dictionary FST
    void fill_hotword_dictionary();
    bool add_word_to_hotword_dictionary(const std::vector<std::string>& characters,
                                        fst::StdVectorFst* dictionary,
                                        float weight);

private:
    /* data */
    size_t dict_size_;
    int SPACE_ID_;
    const std::vector<std::string>& vocabulary;
    std::unordered_map<std::string, int> char_map_;
    bool is_bpe_based_;
    std::string delimiter_;
    std::unordered_map<std::string, float> hotword_weight_map_;
};

#endif // HOTWORD_SCORER_H_
