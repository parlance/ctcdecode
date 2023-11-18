#include "path_trie.h"

#include "decoder_utils.h"

PathTrie::PathTrie()
{
    log_prob_b_prev = -NUM_FLT_INF;
    log_prob_nb_prev = -NUM_FLT_INF;
    log_prob_b_cur = -NUM_FLT_INF;
    log_prob_nb_cur = -NUM_FLT_INF;

    log_prob_b_prev_hw = -NUM_FLT_INF;
    log_prob_nb_prev_hw = -NUM_FLT_INF;
    log_prob_b_cur_hw = -NUM_FLT_INF;
    log_prob_nb_cur_hw = -NUM_FLT_INF;

    log_prob_c = -NUM_FLT_INF;
    score = -NUM_FLT_INF;
    score_hw = -NUM_FLT_INF;

    ROOT_ = -1;
    character = ROOT_;
    timestep = 0;
    exists_ = true;
    parent = nullptr;
    is_hotpath_ = false;
    hotword_score = 0.0;
    shortest_unigram_length = 0;
    hotword_weight = 0.0;
    partial_hotword = "";

    lexicon_ = nullptr;
    lexicon_state_ = 0;
    has_lexicon_ = false;
    is_word_start_char_ = false;

    matcher_ = nullptr;
    hotword_matcher = nullptr;
    hotword_dictionary_state = 0;
    hotword_match_len = 0;
}

PathTrie::~PathTrie()
{
    for (auto child : children_) {
        delete child.second;
    }
}

PathTrie* PathTrie::get_path_trie(int new_char,
                                  int new_timestep,
                                  float cur_log_prob_c,
                                  bool reset,
                                  bool check_lexicon)
{
    auto child = children_.begin();
    for (child = children_.begin(); child != children_.end(); ++child) {
        if (child->first == new_char) {
            if (child->second->log_prob_c < cur_log_prob_c) {
                child->second->log_prob_c = cur_log_prob_c;
                child->second->timestep = new_timestep;
            }
            break;
        }
    }
    if (child != children_.end()) {
        if (!child->second->exists_) {
            child->second->exists_ = true;
            child->second->log_prob_b_prev = -NUM_FLT_INF;
            child->second->log_prob_nb_prev = -NUM_FLT_INF;
            child->second->log_prob_b_cur = -NUM_FLT_INF;
            child->second->log_prob_nb_cur = -NUM_FLT_INF;
            child->second->log_prob_b_prev_hw = -NUM_FLT_INF;
            child->second->log_prob_nb_prev_hw = -NUM_FLT_INF;
            child->second->log_prob_b_cur_hw = -NUM_FLT_INF;
            child->second->log_prob_nb_cur_hw = -NUM_FLT_INF;
            child->second->hotword_matcher = hotword_matcher;
        }
        return (child->second);
    } else {
        if (has_lexicon_ && check_lexicon) {
            matcher_->SetState(lexicon_state_);
            bool found = matcher_->Find(new_char + 1);
            if (!found) {
                // Adding this character causes word outside
                //  lexicon
                auto FSTZERO = fst::TropicalWeight::Zero();
                auto final_weight = lexicon_->Final(lexicon_state_);
                bool is_final = (final_weight != FSTZERO);
                if (is_final && reset) {
                    lexicon_state_ = lexicon_->Start();
                }
                return nullptr;
            } else {

                PathTrie* new_path = create_new_node(new_char, new_timestep, cur_log_prob_c);
                // set spell checker state
                // check to see if next state is final
                auto FSTZERO = fst::TropicalWeight::Zero();
                auto final_weight = lexicon_->Final(matcher_->Value().nextstate);
                bool is_final = (final_weight != FSTZERO);
                if (is_final && reset) {
                    // restart spell checker at the start state
                    new_path->lexicon_state_ = lexicon_->Start();
                } else {
                    // go to next state
                    new_path->lexicon_state_ = matcher_->Value().nextstate;
                }

                children_.push_back(std::make_pair(new_char, new_path));
                return new_path;
            }
        } else {
            PathTrie* new_path = create_new_node(new_char, new_timestep, cur_log_prob_c);
            children_.push_back(std::make_pair(new_char, new_path));
            return new_path;
        }
    }
}

/**
 * @brief Creates new PathTrie node with the given character, timestep and log prob
 *
 * @param new_char, character id
 * @param new_timestep, timestep
 * @param cur_log_prob_c, character probability at this timestep
 * @param has_lexicon, if lexicon is set
 */
PathTrie* PathTrie::create_new_node(int new_char, int new_timestep, float cur_log_prob_c)
{
    PathTrie* new_path = new PathTrie;

    new_path->character = new_char;
    new_path->timestep = new_timestep;
    new_path->parent = this;
    new_path->log_prob_c = cur_log_prob_c;
    new_path->hotword_matcher = hotword_matcher;
    new_path->hotword_dictionary_state = hotword_dictionary_state;
    new_path->hotword_match_len = hotword_match_len;
    new_path->shortest_unigram_length = shortest_unigram_length;
    new_path->hotword_weight = hotword_weight;
    new_path->partial_hotword = partial_hotword;

    if (has_lexicon_) {
        new_path->lexicon_ = lexicon_;
        new_path->has_lexicon_ = true;
        new_path->matcher_ = matcher_;
    }

    return new_path;
}

PathTrie* PathTrie::get_path_vec(std::vector<int>& output, std::vector<int>& timesteps)
{
    return get_path_vec(output, timesteps, ROOT_);
}

PathTrie* PathTrie::get_path_vec(std::vector<int>& output,
                                 std::vector<int>& timesteps,
                                 int stop,
                                 size_t max_steps)
{
    if (character == stop || character == ROOT_ || output.size() == max_steps) {
        std::reverse(output.begin(), output.end());
        std::reverse(timesteps.begin(), timesteps.end());
        return this;
    } else {
        output.push_back(character);
        timesteps.push_back(timestep);
        return parent->get_path_vec(output, timesteps, stop, max_steps);
    }
}

void PathTrie::iterate_to_vec(std::vector<PathTrie*>& output)
{
    if (exists_) {

        log_prob_b_prev = log_prob_b_cur;
        log_prob_nb_prev = log_prob_nb_cur;

        log_prob_b_prev_hw = log_prob_b_cur_hw;
        log_prob_nb_prev_hw = log_prob_nb_cur_hw;

        score = log_sum_exp(log_prob_b_prev, log_prob_nb_prev);
        score_hw = log_sum_exp(log_prob_b_prev_hw, log_prob_nb_prev_hw);

        log_prob_b_cur = -NUM_FLT_INF;
        log_prob_nb_cur = -NUM_FLT_INF;
        log_prob_b_cur_hw = -NUM_FLT_INF;
        log_prob_nb_cur_hw = -NUM_FLT_INF;

        output.push_back(this);
    }
    for (auto child : children_) {
        child.second->iterate_to_vec(output);
    }
}

void PathTrie::remove()
{
    exists_ = false;

    if (children_.size() == 0) {
        auto child = parent->children_.begin();
        for (child = parent->children_.begin(); child != parent->children_.end(); ++child) {
            if (child->first == character) {
                parent->children_.erase(child);
                break;
            }
        }

        if (parent->children_.size() == 0 && !parent->exists_) {
            parent->remove();
        }

        delete this;
    }
}

void PathTrie::set_lexicon(fst::StdVectorFst* lexicon)
{
    lexicon_ = lexicon;
    lexicon_state_ = lexicon->Start();
    has_lexicon_ = true;
}

using FSTMATCH = fst::SortedMatcher<fst::StdVectorFst>;
void PathTrie::set_matcher(std::shared_ptr<FSTMATCH> matcher) { matcher_ = matcher; }

/**
 * @brief Copies parent's hotword related params to the current node
 */
void PathTrie::copy_parent_hotword_params()
{
    hotword_match_len = parent->hotword_match_len;
    hotword_dictionary_state = parent->hotword_dictionary_state;
    shortest_unigram_length = parent->shortest_unigram_length;
    hotword_weight = parent->hotword_weight;
    partial_hotword = parent->partial_hotword;
}

/**
 * @brief Resets the hotword related params of the current node
 */
void PathTrie::reset_hotword_params()
{
    hotword_match_len = 0;
    hotword_dictionary_state = 0;
    shortest_unigram_length = 0;
    hotword_weight = 0.0;
    partial_hotword = "";
}

/**
 * @brief Checks if the current node forms OOV word and accordingly updates its
 *  lexicon state
 *
 * @param true, if current node forms OOV word
 * @param false, if current node doesn't form OOV word
 */
bool PathTrie::is_oov_token()
{

    if (has_lexicon_) {

        fst::StdVectorFst::StateId lexicon_state;

        // If this is the start token of the word, then set the lexicon state
        // to the start state of the lexicon, else
        // use the parent's lexicon state
        if (is_word_start_char_) {
            lexicon_state = lexicon_->Start();

        } else {
            lexicon_state = parent->lexicon_state_;
        }

        // check if the character can be extended from the
        // lexicon state
        matcher_->SetState(lexicon_state);
        bool found = matcher_->Find(character + 1);

        // If the character can be extended, then update the lexicon state
        // of the current node to the next state of the matcher, else
        // reset the lexicon state of the current node to the start state
        if (found) {
            lexicon_state_ = matcher_->Value().nextstate;

        } else {
            lexicon_state_ = lexicon_->Start();
        }

        return !found;
    }

    return false;
}