//
//  main.cpp
//  ctcdecode
//
//  Created by Ryan Leary on 10/26/17.
//  Copyright © 2017 Ryan Leary. All rights reserved.
//

#include <iostream>
#include "ctc_beam_entry.h"
#include "ctc_beam_scorer.h"
#include "ctc_beam_search.h"
#include "ctc_decoder.h"
#include "util/status.h"
#include "ctc_test_data.h"

using namespace pytorch;

int main(int argc, const char * argv[]) {
    int beam_width = 25;
    int64_t top_paths = 2;
    
    // set up seq_len array
    int *sequence_lengths = new int[batch_size]{(int)timesteps};
    Eigen::Map<const Eigen::ArrayXi> seq_len(&sequence_lengths[0], batch_size);

    // set up plain beam scorer
    ctc::CTCBeamSearchDecoder<>::DefaultBeamScorer *beam_scorer = new ctc::CTCBeamSearchDecoder<>::DefaultBeamScorer();
    
    // initialize decoder
    ctc::CTCBeamSearchDecoder<> *decoder = new ctc::CTCBeamSearchDecoder<>
    (num_classes, beam_width, beam_scorer, blank_index);

    // set up input array (3d float array is already log probs)
    std::vector<Eigen::Map<Eigen::MatrixXf, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>> inputs;
    inputs.reserve(timesteps);
    float* probs_ptr = &test_rnn_out[0][0][0];
    ptrdiff_t probs_offset = 0;
    const int64_t probs_stride_0 = num_classes*batch_size;
    const int64_t probs_stride_1 = batch_size;
    const int64_t probs_stride_2 = 1;
    for (int t = 0; t < timesteps; ++t) {
        inputs.emplace_back(probs_ptr + probs_offset + (t*probs_stride_0), batch_size, num_classes, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(probs_stride_2, probs_stride_1));
    }

    // Prepare containers for output and scores.
    std::vector<ctc::CTCDecoder::Output> outputs(top_paths);
    for (ctc::CTCDecoder::Output& output : outputs) {
        output.resize(batch_size);
    }
    std::vector<ctc::CTCDecoder::Output> alignments(top_paths);
    for (ctc::CTCDecoder::Output& alignment : alignments) {
        alignment.resize(batch_size);
    }
    std::vector<ctc::CTCDecoder::CharProbability> char_probs(top_paths);
    for (ctc::CTCDecoder::CharProbability& char_ : char_probs) {
      char_.resize(batch_size);
    }
    
    // Prep score matrix for output scores
    float score[1][1] = {{0.0}};
    Eigen::Map<Eigen::MatrixXf> scores(&score[0][0], batch_size, top_paths);

    // run the decode
    ctc::Status stat = decoder->Decode(seq_len, inputs, &outputs, &scores, &alignments, &char_probs);
    std::cout << "decoder return status ok: " << stat.ok() << std::endl;

    std::vector<float> log_probs;
    for (int p=0; p < top_paths; ++p) {
        std::cout << "path " << p << std::endl;
        for (int b=0; b < batch_size; ++b) {
            std::cout << "  batch " << b << "; score: " << scores(b, p) <<std::endl;
            auto& p_batch = outputs[p][b];
            std::cout << "    Result: ";
            for (int64_t t=0; t<p_batch.size(); ++t) {
                std::cout << labels[p_batch[t]];
            }
            auto& alignment_batch = alignments[p][b];
            std::cout << std::endl << "    Alignment:" << std::endl << "    ";
            for (int64_t t=0; t < alignment_batch.size(); ++t) {
                std::cout << alignment_batch[t] << ", ";
            }
            std::cout << std::endl;
        }
    }

    return 0;
}
