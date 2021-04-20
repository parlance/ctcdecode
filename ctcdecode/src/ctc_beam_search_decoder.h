#ifndef CTC_BEAM_SEARCH_DECODER_H_
#define CTC_BEAM_SEARCH_DECODER_H_

#include <string>
#include <utility>
#include <vector>

#include "path_trie.h"
#include "output.h"

/* CTC Beam Search Decoder

 * Parameters:
 *     probs_seq: 2-D vector that each element is a vector of probabilities
 *               over vocabulary of one time step.
 *     vocabulary: A vector of vocabulary.
 *     beam_size: The width of beam search.
 *     cutoff_prob: Cutoff probability for pruning.
 *     cutoff_top_n: Cutoff number for pruning.
 * Return:
 *     A vector that each element is a pair of score  and decoding result,
 *     in desending order.
*/

std::vector<std::pair<double, Output>> ctc_beam_search_decoder(
    const std::vector<std::vector<double>> &probs_seq,
    const std::vector<std::string> &vocabulary,
    size_t beam_size,
    double cutoff_prob = 1.0,
    size_t cutoff_top_n = 40,
    size_t blank_id = 0,
    int log_input = 0);

/* CTC Beam Search Decoder for batch data

 * Parameters:
 *     probs_seq: 3-D vector that each element is a 2-D vector that can be used
 *                by ctc_beam_search_decoder().
 *     vocabulary: A vector of vocabulary.
 *     beam_size: The width of beam search.
 *     num_processes: Number of threads for beam search.
 *     cutoff_prob: Cutoff probability for pruning.
 *     cutoff_top_n: Cutoff number for pruning.
 * Return:
 *     A 2-D vector that each element is a vector of beam search decoding
 *     result for one audio sample.
*/
std::vector<std::vector<std::pair<double, Output>>>
ctc_beam_search_decoder_batch(
    const std::vector<std::vector<std::vector<double>>> &probs_split,
    const std::vector<std::string> &vocabulary,
    size_t beam_size,
    size_t num_processes,
    double cutoff_prob = 1.0,
    size_t cutoff_top_n = 40,
    size_t blank_id = 0,
    int log_input = 0);

class DecoderState
{
  int abs_time_step;
  int space_id;
  size_t beam_size;
  double cutoff_prob;
  size_t cutoff_top_n;
  size_t blank_id;
  int log_input;
  std::vector<std::string> vocabulary;

  std::vector<PathTrie*> prefixes;
  PathTrie root;

public:
  /* Initialize CTC beam search decoder for streaming
   *
   * Parameters:
   *     vocabulary: A vector of vocabulary.
   *     beam_size: The width of beam search.
   *     cutoff_prob: Cutoff probability for pruning.
   *     cutoff_top_n: Cutoff number for pruning.
  */
  DecoderState(const std::vector<std::string> &vocabulary,
               size_t beam_size,
               double cutoff_prob,
               size_t cutoff_top_n,
               size_t blank_id,
               int log_input);
  ~DecoderState() = default;

  /* Process logits in decoder stream
   *
   * Parameters:
   *     probs: 2-D vector where each element is a vector of probabilities
   *               over alphabet of one time step.
  */
  void next(const std::vector<std::vector<double>> &probs_seq);

  /* Get current transcription from the decoder stream state
   *
   * Return:
   *     A vector where each element is a pair of score and decoding result,
   *     in descending order.
  */
  std::vector<std::pair<double, Output>> decode() const;
};

#endif  // CTC_BEAM_SEARCH_DECODER_H_
