
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
      int ctc_beam_decode(THFloatTensor *th_probs, THIntTensor *th_seq_len, THIntTensor *th_output,
                          THFloatTensor *th_scores, int top_paths, int beam_width, int merge_repeated)
      {
        const int64_t max_time = THFloatTensor_size(th_probs, 0);
        const int64_t batch_size = THFloatTensor_size(th_probs, 1);
        const int64_t num_classes = THFloatTensor_size(th_probs, 2);

        // convert tensors to something the beam scorer can use
        // sequence length
        int* seq_len_ptr = THIntTensor_data(th_seq_len);
        ptrdiff_t seq_len_offset = THIntTensor_storageOffset(th_seq_len);
        Eigen::Map<const Eigen::ArrayXi> seq_len(seq_len_ptr + seq_len_offset, batch_size);

        // input logits
        float* probs_ptr = THFloatTensor_data(th_probs);
        ptrdiff_t probs_offset = THFloatTensor_storageOffset(th_probs);
        const int64_t probs_stride_0 = THFloatTensor_stride(th_probs, 0);
        std::vector<Eigen::Map<const Eigen::MatrixXf>> inputs;
        for (int t=0; t < max_time; ++t) {
          inputs.emplace_back(probs_ptr + probs_offset + (t*probs_stride_0), batch_size, num_classes);
        }

        // prepare/initialize output variables
        // paths, batches, class
        std::vector<ctc::CTCDecoder::Output> outputs(top_paths);
        for (ctc::CTCDecoder::Output& output : outputs) {
          output.resize(batch_size);
        }

        float score[batch_size][top_paths] = {{0.0}};
        Eigen::Map<Eigen::MatrixXf> scores(&score[0][0], batch_size, beam_width);

        // initialize beam scorer (TODO: make this extensible)
        ctc::CTCBeamSearchDecoder<>::DefaultBeamScorer beam_scorer;

        // initialize beam search class
        ctc::CTCBeamSearchDecoder<> beam_search(num_classes, beam_width, &beam_scorer,
                                                batch_size, merge_repeated == 1);

        ctc::Status stat = beam_search.Decode(seq_len, inputs, &outputs, &scores);
        if (!stat.ok()) {
          return 0;
        }
        std::vector<float> log_probs;

        for (int p=0; p < top_paths; ++p) {
          int64_t max_decoded = 0;
          int64_t offset = 0;
          for (int b=0; b < batch_size; ++b) {
            auto& p_batch = outputs[p][b];
            int64_t num_decoded = p_batch.size();

            max_decoded = std::max(max_decoded, num_decoded);
            for (int64_t t=0; t < num_decoded; ++t) {
              // TODO: this could be more efficient (significant pointer arithmetic every time currently)
              THIntTensor_set3d(th_output, p, b, t, p_batch[t]);
              THFloatTensor_set2d(th_scores, p, b, scores(b, p));
            }
            for (int64_t t = num_decoded; t < max_time; ++t) {
              THIntTensor_set3d(th_output, p, b, t, -1);
            }
          }
        }
        return 1;
      }
  }
}
