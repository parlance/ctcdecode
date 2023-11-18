#ifndef CTC_BEAM_SEARCH_DECODER_H_
#define CTC_BEAM_SEARCH_DECODER_H_

#include <utility>
#include <vector>

#include "decoder_options.h"
#include "hotword_scorer.h"
#include "output.h"
#include "scorer.h"

/* CTC Beam Search Decoder

 * Parameters:
 *     probs_seq: 2-D vector that each element is a vector of probabilities
 *               over vocabulary of one time step.
 *     DecoderOptions: Contains vocabulary, beam width, cutoff_top_n, cutoff_prob, etc
 *               that are required for beam decoding
 *     ext_scorer: External scorer to evaluate a prefix, which consists of
 *                 n-gram language model scoring and word insertion term.
 *                 Default null, decoding the input sample without scorer.
 *     hotword_scorer: External hotword scorer to boost the score for specific
 *                 words. Default null, decoding the input sample without hotword scorer
 * Return:
 *     A vector that each element is a pair of score  and decoding result,
 *     in desending order.
*/

std::vector<std::pair<double, Output>>
ctc_beam_search_decoder(const std::vector<std::vector<double>>& probs_seq,
                        DecoderOptions* options,
                        Scorer* ext_scorer = nullptr,
                        HotwordScorer* hotword_scorer = nullptr);

/* CTC Beam Search Decoder for batch data

 * Parameters:
 *     probs_seq: 3-D vector that each element is a 2-D vector that can be used
 *                by ctc_beam_search_decoder().
 *     DecoderOptions: Contains vocabulary, beam width, cutoff_top_n, cutoff_prob, etc
 *                 that are required for beam decoding
 *     ext_scorer: External scorer to evaluate a prefix, which consists of
 *                 n-gram language model scoring and word insertion term.
 *                 Default null, decoding the input sample without scorer.
 *     hotword_scorer: External hotword scorer to boost the score for specific
 *                     words. Default null, decoding the input sample without hotword scorer
 * Return:
 *     A 2-D vector that each element is a vector of beam search decoding
 *     result for one audio sample.
*/
std::vector<std::vector<std::pair<double, Output>>>
ctc_beam_search_decoder_batch(const std::vector<std::vector<std::vector<double>>>& probs_split,
                              DecoderOptions* options,
                              Scorer* ext_scorer = nullptr,
                              HotwordScorer* hotword_scorer = nullptr);

class DecoderState {
    int abs_time_step;
    int space_id;
    int apostrophe_id;
    DecoderOptions* options;
    Scorer* ext_scorer;
    HotwordScorer* hotword_scorer;

    std::vector<PathTrie*> prefixes;
    PathTrie root;

public:
    /* Initialize CTC beam search decoder for streaming
     *
     * Parameters:
     *     DecoderOptions: Contains vocabulary, beam width, cutoff_top_n, cutoff_prob, etc
     *                    that are required for beam decoding
     *     ext_scorer: External scorer to evaluate a prefix, which consists of
     *                 n-gram language model scoring and word insertion term.
     *                 Default null, decoding the input sample without scorer.
     *     hotword_scorer: External hotword scorer to boost the score for specific
     *                  words. Default null, decoding the input sample without hotword scorer
     */
    DecoderState(DecoderOptions* options, Scorer* ext_scorer, HotwordScorer* hotword_scorer);
    ~DecoderState() = default;

    /* Process logits in decoder stream
     *
     * Parameters:
     *     probs: 2-D vector where each element is a vector of probabilities
     *               over alphabet of one time step.
     */
    void next(const std::vector<std::vector<double>>& probs_seq);

    bool is_start_of_word(PathTrie* path);

    void update_score(PathTrie* path, float log_prob_c, float lm_score, bool reset_score);

    /* Get current transcription from the decoder stream state
     *
     * Return:
     *     A vector where each element is a pair of score and decoding result,
     *     in descending order.
     */
    std::vector<std::pair<double, Output>> decode();
};

std::vector<std::vector<std::pair<double, Output>>> ctc_beam_search_decoder_batch_with_states(
    const std::vector<std::vector<std::vector<double>>>& probs_split,
    size_t num_processes,
    std::vector<void*>& states,
    const std::vector<bool>& is_eos_s);

#endif // CTC_BEAM_SEARCH_DECODER_H_
