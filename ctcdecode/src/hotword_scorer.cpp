#include "hotword_scorer.h"
#include "decoder_utils.h"

/**
 * @brief Initializes the vocabulary list, hotwords, hotword weights and creates the hotword FST
 */
HotwordScorer::HotwordScorer(const std::vector<std::string>& vocab_list,
                             const std::vector<std::vector<std::string>>& hotwords,
                             const std::vector<float>& hotword_weights,
                             char token_separator,
                             bool is_bpe_based)
    : vocabulary(vocab_list)
{

    this->hotword_weights = hotword_weights;
    this->hotwords = hotwords;

    dictionary = nullptr;

    dict_size_ = 0;

    SPACE_ID_ = -1;
    is_bpe_based_ = is_bpe_based;
    FSTZERO = fst::TropicalWeight::Zero();
    delimiter_ = "$$";
    token_separator = token_separator;
    setup(vocab_list);
}

/**
 * @brief Deletes the hotword FST of the object
 *
 */
HotwordScorer::~HotwordScorer()
{
    if (dictionary != nullptr) {
        delete dictionary;
    }
}

/**
 * @brief To map characters with it's index and create hotword dictionary
 *
 * @param vocab_list, list of labels provided
 */
void HotwordScorer::setup(const std::vector<std::string>& vocab_list)
{
    set_char_map(vocab_list, char_map_, SPACE_ID_);
    fill_hotword_dictionary();
}

/**
 * @brief To add single word to the hotword dictionary (FST). This method also maps
 * the hotword and it's corresponding weight in `hotword_weight_map_` hash map
 *
 * @param characters, list of characters/tokens of the word
 * @param dictionary, FST dictionary to which characters needs to be added
 * @param weight, hotword weight value for the given hotword
 * @return true, when added successfully.
 * @return false, when not added successfully
 */
bool HotwordScorer::add_word_to_hotword_dictionary(const std::vector<std::string>& characters,
                                                   fst::StdVectorFst* dictionary,
                                                   float weight)
{

    std::vector<int> int_word;
    std::string hotword = "";

    for (auto& c : characters) {
        if (c == " ") {
            int_word.push_back(SPACE_ID_ + 1);
            hotword += std::to_string(SPACE_ID_ + 1) + delimiter_;
        } else {
            auto int_c = char_map_.find(c);
            if (int_c != char_map_.end()) {
                int_word.push_back(int_c->second);
                hotword += std::to_string(int_c->second) + delimiter_;
            } else {
                return false; // return without adding
            }
        }
    }

    hotword_weight_map_[hotword] = weight;
    // add word to dictionary
    add_word_to_fst(int_word, dictionary);
    return true; // return with successful adding
}

/**
 * @brief Creates a FST for the hotwords provided
 *
 */
void HotwordScorer::fill_hotword_dictionary()
{

    fst::StdVectorFst dictionary;
    // For each unigram convert to ints and put in trie
    int dict_size = 0;
    int i = 0;
    for (const auto& char_list : hotwords) {
        bool added = add_word_to_hotword_dictionary(char_list, &dictionary, hotword_weights[i]);
        dict_size += added ? 1 : 0;
        ++i;
    }

    dict_size_ = dict_size;

    /* Simplify FST

     * This gets rid of "epsilon" transitions in the FST.
     * These are transitions that don't require a string input to be taken.
     * Getting rid of them is necessary to make the FST determinisitc, but
     * can greatly increase the size of the FST
     */
    fst::RmEpsilon(&dictionary);
    fst::StdVectorFst* new_dict = new fst::StdVectorFst;

    // /* This makes the FST deterministic, meaning for any string input there's
    //  * only one possible state the FST could be in.  It is assumed our
    //  * dictionary is deterministic when using it.
    //  * (lest we'd have to check for multiple transitions at each state)
    //  */
    fst::Determinize(dictionary, new_dict);

    // /* Finds the simplest equivalent fst. This is unnecessary but decreases
    //  * memory usage of the dictionary
    //  */
    fst::Minimize(new_dict);
    this->dictionary = new_dict;
}

/**
 * @brief This methods returns true when the current node can be extended from
 * the given hotword FST dictionary state.
 *
 * @param path, PathTrie node
 * @param dict_state, FST state id
 * @return true, if the current node's character can be extended from the given FST state
 * @return false, if the current node's character cannot be extended from the given FST state
 */
bool HotwordScorer::is_char_extendable_from_state(PathTrie* path,
                                                  fst::StdVectorFst::StateId dict_state)
{
    path->hotword_matcher->SetState(dict_state);
    return path->hotword_matcher->Find(path->character + 1);
}

/**
 * @brief This methods returns true if the given node can form hotword. If true then the hotword can
 * either be extended from the parent node or the current node starts the hotword.
 *
 * @param path, PathTrie node
 * @param space_id, space id from vocabulary list
 * @param apostrophe_id, apostrophe id from vocabulary list
 * @return true, if the path forms a hotword
 * @return false, if the path doesn't form a hotword
 */
bool HotwordScorer::is_hotpath(PathTrie* path, int space_id, int apostrophe_id)
{
    bool is_hotpath_ = path->parent->is_hotpath();

    is_hotpath_ &= is_char_extendable_from_state(path, path->parent->hotword_dictionary_state);

    if (!is_hotpath_ && path->is_word_start_char()) {
        path->reset_hotword_params();
        is_hotpath_ = is_char_extendable_from_state(path, 0);
    }

    return is_hotpath_;
}

/**
 * @brief Finds the shortest possible candidate hotword for the current node
 *
 * @param path, PathTrie node
 * @return candidate hotword and it's length
 */
std::tuple<std::string, int> HotwordScorer::find_shortest_candidate_hotword_length(PathTrie* path)
{
    bool is_final = false;

    int len = path->hotword_match_len;
    fst::StdVectorFst::StateId matcher_state = path->hotword_dictionary_state;

    std::string hotword = path->partial_hotword;

    // loop until final state is reached
    while (!is_final) {
        path->hotword_matcher->SetState(matcher_state);

        auto final_weight = dictionary->Final(matcher_state);
        is_final = (final_weight != FSTZERO);

        if (!is_final) {
            ++len;
            // go to next state
            matcher_state = path->hotword_matcher->Value().nextstate;
            hotword += std::to_string(path->hotword_matcher->Value().ilabel) + delimiter_;
        }
    }

    return std::make_tuple(hotword, len);
}

/**
 * @brief This method updates the current node's hotword param values such as hotword match length,
 * hotword dictionary state, partial_hotword, hotword weight and shortest unigram length and
 * computes the hotword score
 *
 * @param path, PathTrie node
 */
void HotwordScorer::estimate_hw_score(PathTrie* path)
{
    // update state and match length
    path->hotword_match_len += 1;
    path->hotword_dictionary_state = path->hotword_matcher->Value().nextstate;

    // update partial hotword
    path->partial_hotword
        = path->partial_hotword + std::to_string(path->character + 1) + delimiter_;

    // update shortest unigram length, hotword_weight
    int candidate_hotword_length;
    std::string candidate_hotword;

    std::tie(candidate_hotword, candidate_hotword_length)
        = find_shortest_candidate_hotword_length(path);

    path->shortest_unigram_length = candidate_hotword_length;
    path->hotword_weight = hotword_weight_map_[candidate_hotword];

    // calculate hotword score
    path->hotword_score = (path->hotword_weight * (float)(path->hotword_match_len))
                          / (float)(path->shortest_unigram_length);
}