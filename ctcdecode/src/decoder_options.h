#ifndef DECODER_OPTIONS_H
#define DECODER_OPTIONS_H

#include <string>

class DecoderOptions {
public:
    /* Initialize DecoderOptions for CTC beam decoding
     *
     * Parameters:
     *      vocab: A vector of vocabulary (labels).
     *      cutoff_top_n: Cutoff number in pruning. Only the top cutoff_top_n characters
                with the highest probability in the vocab will be used in beam search.
     *      cutoff_prob: Cutoff probability in pruning. 1.0 means no pruning.
     *      beam_width: This controls how broad the beam search is. Higher values are more
                likely to find top beams, but they also will make your beam search exponentially
                slower.
     *      num_processes: Parallelize the batch using num_processes workers.
     *      blank_id: Index of the CTC blank token used when training your
                model.
     *      log_probs_input (bool): False if the model has passed through a softmax and output
                probabilities sum to 1.
     *      is_bpe_based (bool): True if the labels contains bpe tokens else False
     *      unk_score (float): Extra score to be added when an unknown word forms ( default = '-5' )
     *      token_separator (char): prefix of the bpe tokens ( default = '#' )
     */
    DecoderOptions(std::vector<std::string> vocab,
                   size_t cutoff_top_n,
                   double cutoff_prob,
                   size_t beam_width,
                   size_t num_processes,
                   size_t blank_id,
                   bool log_probs_input,
                   bool is_bpe_based,
                   float unk_score,
                   char token_separator)
        : vocab(vocab)
        , cutoff_top_n(cutoff_top_n)
        , cutoff_prob(cutoff_prob)
        , beam_width(beam_width)
        , num_processes(num_processes)
        , blank_id(blank_id)
        , log_probs_input(log_probs_input)
        , is_bpe_based(is_bpe_based)
        , unk_score(unk_score)
        , token_separator(token_separator)
    {
    }

    /* Initialize DecoderOptions with vocabulary alone
     *
     * Parameters:
     *      vocab: A vector of vocabulary (labels).
     */
    DecoderOptions(std::vector<std::string> vocab)
        : vocab(vocab)
    {
    }
    ~DecoderOptions() = default;

    std::vector<std::string> vocab;
    size_t beam_width = 100;
    size_t cutoff_top_n = 40;
    double cutoff_prob = 1.0;
    size_t num_processes = 4;
    size_t blank_id = 0;
    bool log_probs_input = false;
    bool is_bpe_based = false;
    float unk_score = -5;
    char token_separator = '#';
};

#endif // DECODER_OPTIONS_H
