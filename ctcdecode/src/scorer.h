#ifndef SCORER_H_
#define SCORER_H_

#include <string>
#include <unordered_map>

#include "lm/enumerate_vocab.hh"
#include "lm/virtual_interface.hh"
#include "lm/word_index.hh"
#include "util/string_piece.hh"

#include "decoder_utils.h"
#include "path_trie.h"

const double OOV_SCORE = -1000.0;
const std::string START_TOKEN = "<s>";
const std::string UNK_TOKEN = "[UNK]";
const std::string END_TOKEN = "</s>";

enum TokenizerType { CHAR = 0, BPE = 1, WORD = 2 };

static std::map<std::string, TokenizerType> StringToTokenizerType
    = { { "character", TokenizerType::CHAR },
        { "bpe", TokenizerType::BPE },
        { "word", TokenizerType::WORD } };

// Implement a callback to retrive the lexicon of language model.
class RetriveStrEnumerateVocab : public lm::EnumerateVocab {
public:
    RetriveStrEnumerateVocab() { }

    void Add(lm::WordIndex index, const StringPiece& str)
    {
        vocabulary.push_back(std::string(str.data(), str.length()));
    }

    std::vector<std::string> vocabulary;
};

/* External scorer to query score for n-gram or sentence, including language
 * model scoring and word insertion.
 *
 * Example:
 *     Scorer scorer(alpha, beta, "path_of_language_model");
 *     scorer.get_log_cond_prob({ "WORD1", "WORD2", "WORD3" });
 *     scorer.get_sent_log_prob({ "WORD1", "WORD2", "WORD3" });
 */
class Scorer {
public:
    Scorer(double alpha,
           double beta,
           const std::string& lm_path,
           const std::vector<std::string>& vocabulary,
           const std::string& lm_type,
           const std::string& lexicon_fst_path);
    ~Scorer();

    double get_log_cond_prob(const std::vector<std::string>& words);

    double get_sent_log_prob(const std::vector<std::string>& words);

    // return the max order
    size_t get_max_order() const { return max_order_; }

    // return the lexicon size of language model
    size_t get_lexicon_size() const { return dict_size_; }

    // retrun true if the language model is character based
    bool is_character_based() const { return lm_type == TokenizerType::CHAR; }

    // retrun true if the language model is bpe based
    bool is_bpe_based() const { return lm_type == TokenizerType::BPE; }
    // retrun true if the language model is word based
    bool is_word_based() const { return lm_type == TokenizerType::WORD; }

    bool has_lexicon() const { return has_lexicon_; }

    std::unordered_map<std::string, int> get_char_map() { return char_map_; }

    std::vector<std::string> get_char_list() { return char_list_; }

    // reset params alpha & beta
    void reset_params(float alpha, float beta);

    // make ngram for a given prefix
    std::vector<std::string> make_ngram(PathTrie* prefix);

    // trransform the labels in index to the vector of words (word based lm) or
    // the vector of characters (character based lm)
    std::vector<std::string> split_labels(const std::vector<int>& labels);

    // language model weight
    double alpha;
    // word insertion weight
    double beta;

    // Whether the lm is character based, or bpe based, or word based
    TokenizerType lm_type;

    // pointer to the lexicon of FST
    void* lexicon;

protected:
    // necessary setup: load language model, set char map, fill FST's lexicon
    void setup(const std::string& lm_path,
               const std::vector<std::string>& vocab_list,
               const std::string& lexicon_fst_path);

    // load language model from given path
    void load_lm(const std::string& lm_path);

    // fill lexicon for FST
    void load_lexicon(bool add_space, const std::string& lexicon_fst_path);

    // load FST from given path
    void load_lexicon_from_fst_file(const std::string& lexicon_fst_path);

    double get_log_prob(const std::vector<std::string>& words);

    // translate the vector in index to string
    std::string vec2str(const std::vector<int>& input);

private:
    void* language_model_;
    size_t max_order_;
    size_t dict_size_;
    int SPACE_ID_;
    bool has_lexicon_;
    std::vector<std::string> char_list_;
    std::unordered_map<std::string, int> char_map_;

    std::vector<std::string> vocabulary_;
};

#endif // SCORER_H_
