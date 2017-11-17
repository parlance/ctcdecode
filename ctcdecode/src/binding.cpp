#include <iostream>
#include <string>
#include <vector>
#include "TH.h"
#include "scorer.h"
#include "ctc_beam_search_decoder.h"


int beam_decode(THFloatTensor *th_probs,
                const char* labels,
                int vocab_size,
                size_t beam_size,
                size_t num_processes,
                double cutoff_prob,
                size_t cutoff_top_n,
                size_t blank_id,
                void *scorer,
                THIntTensor *th_output,
                THIntTensor *th_scores,
                THIntTensor *th_seq_length)
{
    std::vector<std::string> new_vocab;
    for (int i = 0; i < vocab_size; ++i) {
        new_vocab.push_back(std::string(1, labels[i]));
    }
    Scorer *ext_scorer = NULL;
    if(scorer != NULL){
        ext_scorer = static_cast<Scorer *>(scorer);
    }
    const int64_t max_time = THFloatTensor_size(th_probs, 0);
    const int64_t batch_size = THFloatTensor_size(th_probs, 1);
    const int64_t num_classes = THFloatTensor_size(th_probs, 2);

    // input logits
    std::vector<std::vector<std::vector<double>>> inputs;
    for (int t=0; t < max_time; ++t) {
        std::vector<std::vector<double>> temp (batch_size, std::vector<double>(num_classes));
        for (int b=0; b < batch_size; ++b){
            for (int n=0; n < num_classes; ++n){
                float val = THFloatTensor_get3d(th_probs, t, b, n);
                temp[b][n] = val;
            }
        }
        inputs.push_back(temp);
    }

    std::vector<std::vector<std::pair<double, std::vector<int>>>> batch_results =
    ctc_beam_search_decoder_batch(inputs, new_vocab, beam_size, num_processes, cutoff_prob, cutoff_top_n, blank_id, ext_scorer);

    for (int b = 0; b < batch_results.size(); ++b){
        std::vector<std::pair<double, std::vector<int>>> path_results = batch_results[b];
        for (int p = 0; p < path_results.size();++p){
            std::pair<double, std::vector<int>> n_path_result = path_results[p];
            std::vector<int> output = n_path_result.second;
            for (int t = 0; t < output.size(); ++t){
                THIntTensor_set3d(th_output, b, p, t, output[t]); // fill output tokens
            }
            THIntTensor_set2d(th_scores, b, p, n_path_result.first); // fill path scores
            THIntTensor_set2d(th_seq_length, b, p, output.size());
        }
    }
    return 1;
}


extern "C"
{
#include "binding.h"
        int paddle_beam_decode(THFloatTensor *th_probs,
                               const char* labels,
                               int vocab_size,
                               size_t beam_size,
                               size_t num_processes,
                               double cutoff_prob,
                               size_t cutoff_top_n,
                               size_t blank_id,
                               THIntTensor *th_output,
                               THIntTensor *th_scores,
                               THIntTensor *th_seq_length){

            return beam_decode(th_probs, labels, vocab_size, beam_size, num_processes,
                        cutoff_prob, cutoff_top_n, blank_id,NULL, th_output, th_scores, th_seq_length);
        }

        int paddle_beam_decode_lm(THFloatTensor *th_probs,
                                  const char* labels,
                                  int vocab_size,
                                  size_t beam_size,
                                  size_t num_processes,
                                  double cutoff_prob,
                                  size_t cutoff_top_n,
                                  size_t blank_id,
                                  void *scorer,
                                  THIntTensor *th_output,
                                  THIntTensor *th_scores,
                                  THIntTensor *th_seq_length){

            return beam_decode(th_probs, labels, vocab_size, beam_size, num_processes,
                        cutoff_prob, cutoff_top_n, blank_id,scorer, th_output, th_scores, th_seq_length);
        }


    void* paddle_get_scorer(double alpha,
                            double beta,
                            const char* lm_path,
                            const char* labels,
                            int vocab_size) {
        std::vector<std::string> new_vocab;
        for (int i = 0; i < vocab_size; ++i) {
            new_vocab.push_back(std::string(1, labels[i]));
        }
        Scorer* scorer = new Scorer(alpha, beta, lm_path, new_vocab);
        return static_cast<void*>(scorer);
    }

    int is_character_based(void *scorer){
        Scorer *ext_scorer  = static_cast<Scorer *>(scorer);
        return ext_scorer->is_character_based();
    }
    size_t get_max_order(void *scorer){
        Scorer *ext_scorer  = static_cast<Scorer *>(scorer);
        return ext_scorer->get_max_order();
    }
    size_t get_dict_size(void *scorer){
        Scorer *ext_scorer  = static_cast<Scorer *>(scorer);
        return ext_scorer->get_dict_size();
    }

    void reset_params(void *scorer, double alpha, double beta){
        Scorer *ext_scorer  = static_cast<Scorer *>(scorer);
        ext_scorer->reset_params(alpha, beta);
    }
}
