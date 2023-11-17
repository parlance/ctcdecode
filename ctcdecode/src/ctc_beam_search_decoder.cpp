#include "ctc_beam_search_decoder.h"

#include <cmath>
#include <iostream>
#include <map>

#include "ThreadPool.h"
#include "decoder_utils.h"
#include "fst/fstlib.h"
#include "path_trie.h"

using FSTMATCH = fst::SortedMatcher<fst::StdVectorFst>;

DecoderState::DecoderState(DecoderOptions* options,
                           Scorer* ext_scorer,
                           HotwordScorer* hotword_scorer)
    : abs_time_step(0)
    , options(options)
    , ext_scorer(ext_scorer)
    , hotword_scorer(hotword_scorer)
{
    space_id = -2;
    apostrophe_id = -3;

    // assign space id and apostrophe id if present in vocabulary
    int id = 0;
    for (auto it = options->vocab.begin(); it != options->vocab.end(); ++it) {
        if (*it == " ") {
            space_id = id;
        } else if (*it == "'") {
            apostrophe_id = id;
        }
        ++id;
    }

    // init prefixes' root
    root.score = root.log_prob_b_prev = 0.0;
    root.score_hw = root.log_prob_b_prev_hw = 0.0;
    prefixes.push_back(&root);

    if (ext_scorer != nullptr && ext_scorer->has_lexicon()) {

        auto fst_dict = static_cast<fst::StdVectorFst*>(ext_scorer->lexicon);
        fst::StdVectorFst* dict_ptr = fst_dict->Copy(true);
        root.set_lexicon(dict_ptr);
        auto matcher
            = std::make_shared<fst::SortedMatcher<fst::StdVectorFst>>(*dict_ptr, fst::MATCH_INPUT);
        root.set_matcher(matcher);
    }

    if (hotword_scorer != nullptr) {
        auto hotword_matcher
            = std::make_shared<FSTMATCH>(hotword_scorer->dictionary, fst::MATCH_INPUT);
        root.hotword_matcher = hotword_matcher;
    }
}

/**
 * @brief This methods returns true when the given node can be a start of the word.
 * Supports both bpe and character based labels
 *
 * @param path, PathTrie node
 * @return true, if the current node's character/token can start a word
 * @return false, if the current node's character/token cannot start a word
 */
bool DecoderState::is_start_of_word(PathTrie* path)
{

    bool is_bpe_based_start_token = options->is_bpe_based
                                    && !is_mergeable_bpe_token(options->vocab[path->character],
                                                               path->character,
                                                               path->parent->character,
                                                               apostrophe_id,
                                                               options->token_separator);

    bool is_char_based_start_token
        = !options->is_bpe_based
          && (path->parent->character == space_id || path->parent->character == -1);

    return is_bpe_based_start_token || is_char_based_start_token;
}

/**
 * @brief Updates both original and hotword non-blank scores of the current path node. If the
 current ends a
 * hotword then the original score (log_p) is updated with the actual score (log_p_hw, contains both
 original and hotword scores)
 *
 * @param path, PathTrie node
 * @param log_prob_c, log probablity of the node
 * @param lm_score, language model score for the current node
 * @param reset_score, whether to consider previous node's original score instead of original +
 hotword score . Score resetting happens when partial hotword is formed
 */
void DecoderState::update_score(PathTrie* path, float log_prob_c, float lm_score, bool reset_score)
{
    float log_p_lm_score = log_prob_c + lm_score;

    float log_p = -NUM_FLT_INF;
    float log_p_hw = -NUM_FLT_INF;

    bool is_complete_hotword
        = (path->hotword_match_len > 0 && path->hotword_match_len == path->shortest_unigram_length);

    if (path->character == path->parent->character) {

        if (path->parent->log_prob_b_prev > -NUM_FLT_INF) {
            log_p = log_p_lm_score + path->parent->log_prob_b_prev;

            if (reset_score) {
                // when the current token is not part of hotword whereas prev token
                // is, then consider the original score for the current token
                // scoring
                log_p_hw = log_p + path->hotword_score;
            } else {
                log_p_hw = log_p_lm_score + path->parent->log_prob_b_prev_hw + path->hotword_score;

                if (is_complete_hotword) {
                    // original score needs to be updated with hotword score when
                    // complete hotword is formed.
                    log_p = log_p_hw;
                }
            }
        }

    } else if (path->character != path->parent->character) {
        log_p = log_p_lm_score + path->parent->score;

        if (reset_score) {
            log_p_hw = log_p + path->hotword_score;
        } else {
            log_p_hw = log_p_lm_score + path->parent->score_hw + path->hotword_score;

            if (is_complete_hotword) {
                log_p = log_p_hw;
            }
        }
    }

    path->log_prob_nb_cur = log_sum_exp(path->log_prob_nb_cur, log_p);
    path->log_prob_nb_cur_hw = log_sum_exp(path->log_prob_nb_cur_hw, log_p_hw);
}

void DecoderState::next(const std::vector<std::vector<double>>& probs_seq)
{
    // dimension check
    size_t num_time_steps = probs_seq.size();
    for (size_t i = 0; i < num_time_steps; ++i) {
        VALID_CHECK_EQ(probs_seq[i].size(),
                       options->vocab.size(),
                       "The shape of probs_seq does not match with "
                       "the shape of the vocabulary");
    }

    // prefix search over time
    for (size_t time_step = 0; time_step < num_time_steps; ++time_step, ++abs_time_step) {
        auto& prob = probs_seq[time_step];

        float min_cutoff = -NUM_FLT_INF;
        bool full_beam = false;
        if (ext_scorer != nullptr) {
            size_t num_prefixes = std::min(prefixes.size(), options->beam_width);
            std::sort(prefixes.begin(), prefixes.begin() + num_prefixes, prefix_compare);
            float blank_prob = options->log_probs_input ? prob[options->blank_id]
                                                        : std::log(prob[options->blank_id]);
            min_cutoff = prefixes[num_prefixes - 1]->score_hw + blank_prob
                         - std::max(0.0, ext_scorer->beta);
            full_beam = (num_prefixes == options->beam_width);
        }

        std::vector<std::pair<size_t, float>> log_prob_idx = get_pruned_log_probs(
            prob, options->cutoff_prob, options->cutoff_top_n, options->log_probs_input);

        // loop over chars
        for (size_t index = 0; index < log_prob_idx.size(); ++index) {
            auto c = log_prob_idx[index].first;
            auto log_prob_c = log_prob_idx[index].second;

            for (size_t i = 0; i < prefixes.size() && i < options->beam_width; ++i) {

                auto prefix = prefixes[i];

                if (full_beam && log_prob_c + prefix->score_hw < min_cutoff) {
                    break;
                }
                // blank
                if (c == options->blank_id) {
                    prefix->log_prob_b_cur
                        = log_sum_exp(prefix->log_prob_b_cur, log_prob_c + prefix->score);
                    prefix->log_prob_b_cur_hw
                        = log_sum_exp(prefix->log_prob_b_cur_hw, log_prob_c + prefix->score_hw);
                    continue;
                }

                // repeated character
                if (c == prefix->character) {
                    prefix->log_prob_nb_cur = log_sum_exp(prefix->log_prob_nb_cur,
                                                          log_prob_c + prefix->log_prob_nb_prev);

                    prefix->log_prob_nb_cur_hw = log_sum_exp(
                        prefix->log_prob_nb_cur_hw, log_prob_c + prefix->log_prob_nb_prev_hw);
                }

                // get new prefix
                auto new_path = prefix->get_path_trie(
                    c, abs_time_step, log_prob_c, true, !options->is_bpe_based);

                if (new_path != nullptr) {

                    float lm_score = 0.0;
                    bool is_hotpath = false;
                    bool reset_score = false;

                    // check if the current node is a start of the word
                    if ((ext_scorer != nullptr || hotword_scorer != nullptr)
                        && is_start_of_word(new_path)) {
                        new_path->mark_as_word_start_char();
                    }

                    // check if the current node is part of a hotword
                    if (hotword_scorer != nullptr) {
                        new_path->copy_parent_hotword_params();
                        is_hotpath = hotword_scorer->is_hotpath(new_path, space_id, apostrophe_id);

                        if (!is_hotpath) {
                            new_path->reset_hotword_params();
                            if (prefix->is_hotpath()) {
                                reset_score = true;
                            }
                        }
                    }

                    // hotword scoring
                    if (is_hotpath) {
                        new_path->mark_as_hotpath();

                        // need to consider original score when previous word is a
                        // partial hotword
                        if (prefix->is_hotpath() && new_path->hotword_dictionary_state == 0) {
                            reset_score = true;
                        }

                        // update hotword related params of new node and calculate hotword score
                        hotword_scorer->estimate_hw_score(new_path);
                    }
                    // unknown scoring
                    else {

                        // reset the hotword parameters of the new node
                        if (hotword_scorer != nullptr) {
                            new_path->reset_hotword_params();
                            if (prefix->is_hotpath()) {
                                reset_score = true;
                            }
                        }

                        // check if the current node forms OOV word and add unk score
                        if (options->is_bpe_based && ext_scorer != nullptr
                            && ext_scorer->has_lexicon()) {
                            bool is_oov = new_path->is_oov_token();
                            if (is_oov) {
                                lm_score += options->unk_score;
                            }
                        }
                    }

                    // language model scoring
                    if (ext_scorer != nullptr
                        && (c == space_id || ext_scorer->is_character_based()
                            || ext_scorer->is_bpe_based())) {

                        PathTrie* prefix_to_score = nullptr;
                        // skip scoring the space
                        if (ext_scorer->is_character_based() || ext_scorer->is_bpe_based()) {
                            prefix_to_score = new_path;
                        } else {
                            prefix_to_score = prefix;
                        }
                        std::vector<std::string> ngram;
                        ngram = ext_scorer->make_ngram(prefix_to_score);
                        lm_score += ext_scorer->get_log_cond_prob(ngram) * ext_scorer->alpha;
                        lm_score += ext_scorer->beta;
                    }

                    // update original and hotword score for the new path
                    update_score(new_path, log_prob_c, lm_score, reset_score);
                }

            } // end of loop over prefix
        }     // end of loop over vocabulary

        prefixes.clear();
        // update log probs
        root.iterate_to_vec(prefixes);

        // only preserve top beam_size prefixes
        if (prefixes.size() >= options->beam_width) {
            std::nth_element(prefixes.begin(),
                             prefixes.begin() + options->beam_width,
                             prefixes.end(),
                             prefix_compare);
            for (size_t i = options->beam_width; i < prefixes.size(); ++i) {
                prefixes[i]->remove();
            }

            prefixes.resize(options->beam_width);
        }

    } // end of loop over time
}

std::vector<std::pair<double, Output>> DecoderState::decode()
{
    std::vector<PathTrie*> prefixes_copy = prefixes;
    std::unordered_map<const PathTrie*, float> scores;
    for (PathTrie* prefix : prefixes_copy) {
        scores[prefix] = prefix->score_hw;
    }

    // score the last word of each prefix that doesn't end with space
    if (ext_scorer != nullptr
        && !(ext_scorer->is_character_based() || ext_scorer->is_bpe_based())) {
        for (size_t i = 0; i < options->beam_width && i < prefixes_copy.size(); ++i) {
            auto prefix = prefixes_copy[i];
            if (!prefix->is_empty() && prefix->character != space_id) {
                float score = 0.0;
                std::vector<std::string> ngram = ext_scorer->make_ngram(prefix);
                score = ext_scorer->get_log_cond_prob(ngram) * ext_scorer->alpha;
                score += ext_scorer->beta;
                scores[prefix] += score;
            }
        }
    }

    using namespace std::placeholders;
    size_t num_prefixes = std::min(prefixes_copy.size(), options->beam_width);
    std::sort(prefixes_copy.begin(),
              prefixes_copy.begin() + num_prefixes,
              std::bind(prefix_compare_external_scores, _1, _2, scores));

    // compute aproximate ctc score as the return score, without affecting the
    // return order of decoding result. To delete when decoder gets stable.
    for (size_t i = 0; i < options->beam_width && i < prefixes_copy.size(); ++i) {
        double approx_ctc = scores[prefixes_copy[i]];
        if (ext_scorer != nullptr
            && !(ext_scorer->is_character_based() || ext_scorer->is_bpe_based())) {
            std::vector<int> output;
            std::vector<int> timesteps;
            prefixes_copy[i]->get_path_vec(output, timesteps);
            auto prefix_length = output.size();
            auto words = ext_scorer->split_labels(output);
            // remove word insert
            approx_ctc = approx_ctc - prefix_length * ext_scorer->beta;
            // remove language model weight:
            approx_ctc -= (ext_scorer->get_sent_log_prob(words)) * ext_scorer->alpha;
        }
        prefixes_copy[i]->approx_ctc = approx_ctc;
    }

    return get_beam_search_result(prefixes_copy, options->beam_width);
}

std::vector<std::pair<double, Output>>
ctc_beam_search_decoder(const std::vector<std::vector<double>>& probs_seq,
                        DecoderOptions* options,
                        Scorer* ext_scorer,
                        HotwordScorer* hotword_scorer)
{
    DecoderState state(options, ext_scorer, hotword_scorer);
    state.next(probs_seq);
    return state.decode();
}

std::vector<std::pair<double, Output>>
ctc_beam_search_decoder_with_given_state(const std::vector<std::vector<double>>& probs_seq,
                                         DecoderState* state,
                                         bool is_eos)
{
    state->next(probs_seq);
    if (is_eos) {
        return state->decode();
    } else {
        return {};
    }
}

std::vector<std::vector<std::pair<double, Output>>>
ctc_beam_search_decoder_batch(const std::vector<std::vector<std::vector<double>>>& probs_split,
                              DecoderOptions* options,
                              Scorer* ext_scorer,
                              HotwordScorer* hotword_scorer)
{
    VALID_CHECK_GT(options->num_processes, 0, "num_processes must be nonnegative!");
    // thread pool
    ThreadPool pool(options->num_processes);
    // number of samples
    size_t batch_size = probs_split.size();

    // enqueue the tasks of decoding
    std::vector<std::future<std::vector<std::pair<double, Output>>>> res;
    for (size_t i = 0; i < batch_size; ++i) {
        res.emplace_back(pool.enqueue(ctc_beam_search_decoder,
                                      std::cref(probs_split[i]),
                                      options,
                                      ext_scorer,
                                      hotword_scorer));
    }

    // get decoding results
    std::vector<std::vector<std::pair<double, Output>>> batch_results;
    for (size_t i = 0; i < batch_size; ++i) {
        batch_results.emplace_back(res[i].get());
    }
    return batch_results;
}

std::vector<std::vector<std::pair<double, Output>>> ctc_beam_search_decoder_batch_with_states(
    const std::vector<std::vector<std::vector<double>>>& probs_split,
    size_t num_processes,
    std::vector<void*>& states,
    const std::vector<bool>& is_eos_s)

{
    VALID_CHECK_GT(num_processes, 0, "num_processes must be nonnegative!");
    // thread pool
    ThreadPool pool(num_processes);
    // number of samples
    size_t batch_size = probs_split.size();

    // enqueue the tasks of decoding
    std::vector<std::future<std::vector<std::pair<double, Output>>>> res;
    for (size_t i = 0; i < batch_size; ++i) {
        res.emplace_back(pool.enqueue(ctc_beam_search_decoder_with_given_state,
                                      std::cref(probs_split[i]),
                                      static_cast<DecoderState*>(states[i]),
                                      is_eos_s[i]));
    }

    // get decoding results
    std::vector<std::vector<std::pair<double, Output>>> batch_results;
    for (size_t i = 0; i < batch_size; ++i) {
        batch_results.emplace_back(res[i].get());
    }
    return batch_results;
}
