#include "scorer.h"

#include <iostream>
#include <unistd.h>

#include "lm/config.hh"
#include "lm/model.hh"
#include "lm/state.hh"
#include "util/string_piece.hh"
#include "util/tokenize_piece.hh"

#include "decoder_utils.h"

using namespace lm::ngram;

Scorer::Scorer(double alpha,
               double beta,
               const std::string& lm_path,
               const std::vector<std::string>& vocab_list,
               const std::string& lm_type,
               const std::string& lexicon_fst_path)
{
    this->alpha = alpha;
    this->beta = beta;
    this->lm_type = StringToTokenizerType[lm_type];
    lexicon = nullptr;
    language_model_ = nullptr;
    max_order_ = 0;
    dict_size_ = 0;
    SPACE_ID_ = -1;

    char_list_ = vocab_list;
    setup(lm_path, vocab_list, lexicon_fst_path);
}

Scorer::~Scorer()
{
    if (language_model_ != nullptr) {
        delete static_cast<lm::base::Model*>(language_model_);
    }
    if (lexicon != nullptr) {
        delete static_cast<fst::StdVectorFst*>(lexicon);
    }
}

void Scorer::setup(const std::string& lm_path,
                   const std::vector<std::string>& vocab_list,
                   const std::string& lexicon_fst_path)
{
    // load language model
    load_lm(lm_path);
    // set char map for scorer
    set_char_map(vocab_list, char_map_, SPACE_ID_);
    // fill the dictionary for FST
    if (is_word_based() || !lexicon_fst_path.empty()) {
        load_lexicon(true, lexicon_fst_path);
    }
}

void Scorer::load_lm(const std::string& lm_path)
{
    const char* filename = lm_path.c_str();
    VALID_CHECK_EQ(access(filename, F_OK), 0, "Invalid language model path");

    RetriveStrEnumerateVocab enumerate;
    lm::ngram::Config config;
    config.enumerate_vocab = &enumerate;
    language_model_ = lm::ngram::LoadVirtual(filename, config);
    max_order_ = static_cast<lm::base::Model*>(language_model_)->Order();
    vocabulary_ = enumerate.vocabulary;

    if (!is_bpe_based()) {
        for (auto it = vocabulary_.begin(); it != vocabulary_.end(); ++it) {
            if (is_character_based() && *it != UNK_TOKEN && *it != START_TOKEN && *it != END_TOKEN
                && get_utf8_str_len(*it) > 1) {
                lm_type = TokenizerType::WORD;
                break; // terminate after `lm_type` is set
            }
        }
    }
}

double Scorer::get_log_cond_prob(const std::vector<std::string>& words)
{
    lm::base::Model* model = static_cast<lm::base::Model*>(language_model_);
    double cond_prob;
    lm::ngram::State state, tmp_state, out_state;
    // avoid to inserting <s> in begin
    model->NullContextWrite(&state);
    for (size_t i = 0; i < words.size(); ++i) {
        lm::WordIndex word_index = 0;
        if (words[i] != UNK_TOKEN) {
            word_index = model->BaseVocabulary().Index(words[i]);
        }
        // encounter OOV
        if (word_index == 0) {
            return OOV_SCORE;
        }
        cond_prob = model->BaseScore(&state, word_index, &out_state);
        tmp_state = state;
        state = out_state;
        out_state = tmp_state;
    }
    // return  loge prob
    return cond_prob / NUM_FLT_LOGE;
}

double Scorer::get_sent_log_prob(const std::vector<std::string>& words)
{
    std::vector<std::string> sentence;
    if (words.size() == 0) {
        for (size_t i = 0; i < max_order_; ++i) {
            sentence.push_back(START_TOKEN);
        }
    } else {
        for (size_t i = 0; i < max_order_ - 1; ++i) {
            sentence.push_back(START_TOKEN);
        }
        sentence.insert(sentence.end(), words.begin(), words.end());
    }
    sentence.push_back(END_TOKEN);
    return get_log_prob(sentence);
}

double Scorer::get_log_prob(const std::vector<std::string>& words)
{
    assert(words.size() > max_order_);
    double score = 0.0;
    for (size_t i = 0; i < words.size() - max_order_ + 1; ++i) {
        std::vector<std::string> ngram(words.begin() + i, words.begin() + i + max_order_);
        score += get_log_cond_prob(ngram);
    }
    return score;
}

void Scorer::reset_params(float alpha, float beta)
{
    this->alpha = alpha;
    this->beta = beta;
}

std::string Scorer::vec2str(const std::vector<int>& input)
{
    std::string word;
    for (auto ind : input) {
        word += char_list_[ind];
    }
    return word;
}

std::vector<std::string> Scorer::split_labels(const std::vector<int>& labels)
{
    if (labels.empty())
        return {};

    std::string s = vec2str(labels);
    std::vector<std::string> words;
    if (is_character_based()) {
        words = split_utf8_str(s);
    } else {
        words = split_str(s, " ");
    }
    return words;
}

std::vector<std::string> Scorer::make_ngram(PathTrie* prefix)
{
    std::vector<std::string> ngram;
    PathTrie* current_node = prefix;
    PathTrie* new_node = nullptr;

    for (int order = 0; order < max_order_; ++order) {
        std::vector<int> prefix_vec;
        std::vector<int> prefix_steps;

        if (is_character_based() || is_bpe_based()) {
            new_node = current_node->get_path_vec(prefix_vec, prefix_steps, -1, 1);
            current_node = new_node;
        } else {
            new_node = current_node->get_path_vec(prefix_vec, prefix_steps, SPACE_ID_);
            current_node = new_node->parent; // Skipping spaces
        }

        // reconstruct word
        std::string word = vec2str(prefix_vec);
        ngram.push_back(word);

        if (new_node->character == -1) {
            // No more spaces, but still need order
            for (int i = 0; i < max_order_ - order - 1; ++i) {
                ngram.push_back(START_TOKEN);
            }
            break;
        }
    }
    std::reverse(ngram.begin(), ngram.end());
    return ngram;
}

/**
 * @brief Loads FST from the given path
 *
 * @param lexicon_fst_path, Path to the file containing the FST
 */
void Scorer::load_lexicon_from_fst_file(const std::string& lexicon_fst_path)
{

    auto startTime = std::chrono::high_resolution_clock::now();
    fst::FstReadOptions read_options;
    // Read the FST from the file
    fst::StdVectorFst* dict = fst::StdVectorFst::Read(lexicon_fst_path);
    if (!dict) {
        std::cerr << "Failed to read FST from file: " << lexicon_fst_path << std::endl;
        exit(EXIT_FAILURE);
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration
        = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
    // Convert duration to seconds
    auto seconds = duration / 1000000.0;

    std::cout << "Total time taken for reading the FST file: " << seconds << " seconds"
              << std::endl;

    this->lexicon = dict;
}

/**
 * @brief Creates FST lexicon from the LM vocabulary or from the given FST
 *
 * @param add_space, whether to add space in the dictionary after each word
 * @param lexicon_fst_path, Path to the file containing the FST
 */
void Scorer::load_lexicon(bool add_space, const std::string& lexicon_fst_path)
{
    fst::StdVectorFst lexicon;
    // For each unigram convert to ints and put in trie
    int dict_size = 0;
    has_lexicon_ = true;

    if (lexicon_fst_path.empty()) {
        for (const auto& word : vocabulary_) {
            const auto& characters = split_utf8_str(word);
            bool added
                = add_word_to_lexicon(characters, char_map_, add_space, SPACE_ID_ + 1, &lexicon);
            dict_size += added ? 1 : 0;
        }

        dict_size_ = dict_size;

        /* Simplify FST

         * This gets rid of "epsilon" transitions in the FST.
         * These are transitions that don't require a string input to be taken.
         * Getting rid of them is necessary to make the FST determinisitc, but
         * can greatly increase the size of the FST
         */
        fst::RmEpsilon(&lexicon);
        fst::StdVectorFst* new_lexicon = new fst::StdVectorFst;

        /* This makes the FST deterministic, meaning for any string input there's
         * only one possible state the FST could be in.  It is assumed our
         * dictionary is deterministic when using it.
         * (lest we'd have to check for multiple transitions at each state)
         */
        fst::Determinize(lexicon, new_lexicon);

        /* Finds the simplest equivalent fst. This is unnecessary but decreases
         * memory usage of the dictionary
         */
        fst::Minimize(new_lexicon);
        this->lexicon = new_lexicon;
    } else {
        load_lexicon_from_fst_file(lexicon_fst_path);
    }
}
