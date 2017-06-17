
#include <iostream>
#include "ctc_beam_entry.h"
#include "ctc_beam_scorer.h"
#include "ctc_beam_search.h"
#include "ctc_decoder.h"
#include "util/status.h"
#include "TH.h"

namespace pytorch {
  extern "C"
  {
      int test_ctc_beam_decode() {
        const int batch_size = 1;
        const int timesteps = 5;
        const int top_paths = 3;
        const int num_classes = 6;

        ctc::CTCBeamSearchDecoder<>::DefaultBeamScorer default_scorer;
        ctc::CTCBeamSearchDecoder<> decoder(num_classes, 10 * top_paths, &default_scorer);

        int sequence_lengths[batch_size] = {timesteps};
        float input_data_mat[timesteps][batch_size][num_classes] = {
            {{0, 0.6, 0, 0.4, 0, 0}},
            {{0, 0.5, 0, 0.5, 0, 0}},
            {{0, 0.4, 0, 0.6, 0, 0}},
            {{0, 0.4, 0, 0.6, 0, 0}},
            {{0, 0.4, 0, 0.6, 0, 0}}};

            // The CTCDecoder works with log-probs.
        for (int t = 0; t < timesteps; ++t) {
          for (int b = 0; b < batch_size; ++b) {
            for (int c = 0; c < num_classes; ++c) {
              input_data_mat[t][b][c] = std::log(input_data_mat[t][b][c]);
            }
          }
        }

        // Plain output, without any additional scoring.
        std::vector<ctc::CTCDecoder::Output> expected_output = {
           {{1, 3}, {1, 3, 1}, {3, 1, 3}},
        };

        Eigen::Map<const Eigen::ArrayXi> seq_len(&sequence_lengths[0], batch_size);
        std::cout << "seq_len: " << seq_len << std::endl;
        std::vector<Eigen::Map<const Eigen::MatrixXf>> inputs;
        for (int t = 0; t < timesteps; ++t) {
          inputs.emplace_back(&input_data_mat[t][0][0], batch_size, num_classes);
          std::cout << inputs[t] << std::endl;
        }

        // Prepare containers for output and scores.
        std::vector<ctc::CTCDecoder::Output> outputs(top_paths);
        for (ctc::CTCDecoder::Output& output : outputs) {
          output.resize(batch_size);
        }
        float score[batch_size][top_paths] = {{0.0}};
        Eigen::Map<Eigen::MatrixXf> scores(&score[0][0], batch_size, top_paths);

        decoder.Decode(seq_len, inputs, &outputs, &scores);
      }
      int ctc_beam_decode(THFloatTensor *th_probs, THIntTensor *th_seq_len, THIntTensor *th_output,
                          THFloatTensor *th_scores, int beam_width, int merge_repeated)
      {
        const int64_t max_time = THFloatTensor_size(th_probs, 0);
        const int64_t batch_size = THFloatTensor_size(th_probs, 1);
        const int64_t num_classes = THFloatTensor_size(th_probs, 2);
        const int64_t ndims = THFloatTensor_nDimension(th_probs);
        const int64_t stride0 = THFloatTensor_stride(th_probs,0);
        const int64_t stride1 = THFloatTensor_stride(th_probs,1);
        const int64_t stride2 = THFloatTensor_stride(th_probs,2);

        // TODO: DEBUG, Remove
        std::cout << "Max Timesteps: " << max_time << "\nBatch Size: " << batch_size << "\nNumber of Classes: " << num_classes << std::endl;
        std::cout << "Num dims: " << ndims << "\nStride 0: " << stride0 << "\nStride 1: " << stride1 << "\nStride 2: " << stride2 << std::endl;
        // convert tensors to something the beam scorer can use
        // sequence length
        int* seq_len_ptr = THIntTensor_data(th_seq_len);
        ptrdiff_t seq_len_offset = THIntTensor_storageOffset(th_seq_len);
        Eigen::Map<const Eigen::ArrayXi> seq_len(seq_len_ptr + seq_len_offset, batch_size);
        std::cout << "seq_len: " << seq_len << std::endl;
        std::cout << "STATUS: Converted sequence_length tensor" << std::endl;
        std::cout << "DEBUG: seq_len=" << seq_len << std::endl;

        // input logits
        float* probs_ptr = THFloatTensor_data(th_probs);
        ptrdiff_t probs_offset = THFloatTensor_storageOffset(th_probs);
        std::vector<Eigen::Map<const Eigen::MatrixXf>> inputs;
        for (int t=0; t < max_time; ++t) {
          inputs.emplace_back(probs_ptr + probs_offset + (t*stride0), batch_size, num_classes);
          std::cout << inputs[t] << std::endl;
        }
        std::cout << "STATUS: Converted input logits tensor" << std::endl;
        std::cout << "DEBUG: inputs:\n" << inputs[0] << std::endl;
        // std::cout << "printing flattened input matrix..." << std::endl;
        // for (int t=0; t < max_time; ++t) {
        //   std::cout << t << ": " << th_probs_ptr[t] << std::endl;
        // }

        // prepare/initialize output variables
        int top_paths = 3;
        std::vector<ctc::CTCDecoder::Output> outputs(top_paths);
        for (ctc::CTCDecoder::Output& output : outputs) {
          output.resize(batch_size);
        }
        std::cout << "STATUS: Output char vector created" << std::endl;

        float score[batch_size][beam_width] = {{0.0}};
        Eigen::Map<Eigen::MatrixXf> scores(&score[0][0], batch_size, beam_width);
        std::cout << "STATUS: Output score vector created" << std::endl;

        // initialize beam scorer (TODO: make this extensible)
        ctc::CTCBeamSearchDecoder<>::DefaultBeamScorer beam_scorer;

        // initialize beam search class
        ctc::CTCBeamSearchDecoder<> beam_search(num_classes, beam_width, &beam_scorer,
                                                batch_size, merge_repeated == 1);
        std::cout << "STATUS: beam decoder initialized" << std::endl;

        std::vector<std::vector<std::vector<int> > > best_paths(batch_size);
        beam_search.Decode(seq_len, inputs, &outputs, &scores);
        std::cout << "STATUS: decode executed" << std::endl;
        std::vector<float> log_probs;

        for (int p=0; p < top_paths; ++p) {
          int64_t max_decoded = 0;
          int64_t offset = 0;
          for (int b=0; b < batch_size; ++b) {
            auto& p_batch = outputs[b][p];
            int64_t num_decoded = p_batch.size();
            max_decoded = std::max(max_decoded, num_decoded);
            for (int64_t t=0; t < num_decoded; ++t) {
              std::cout << p_batch[t] << " ";
            }
            std::cout << std::endl;
          }
        }

        //
        // // Assumption: the blank index is num_classes - 1
        // for (int b = 0; b < batch_size; ++b) {
        //   auto& best_paths_b = best_paths[b];
        //   best_paths_b.resize(decode_helper_.GetTopPaths());
        //   for (int t = 0; t < seq_len_t(b); ++t) {
        //     input_chip_t = input_list_t[t].chip(b, 0);
        //     auto input_bi =
        //         Eigen::Map<const Eigen::ArrayXf>(input_chip_t.data(), num_classes);
        //     beam_search.Step(input_bi);
        //   }
        //   OP_REQUIRES_OK(
        //       ctx, beam_search.TopPaths(decode_helper_.GetTopPaths(), &best_paths_b,
        //                                 &log_probs, merge_repeated_));
        //
        //   beam_search.Reset();
        //
        //   for (int bp = 0; bp < decode_helper_.GetTopPaths(); ++bp) {
        //     log_prob_t(b, bp) = log_probs[bp];
        //   }
        // }
        return 1;
      }
  }
}
