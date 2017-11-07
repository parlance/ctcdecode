/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef PYTORCH_CONTRIB_CTC_CTC_BEAM_SEARCH_H_
#define PYTORCH_CONTRIB_CTC_CTC_BEAM_SEARCH_H_

#include <cmath>
#include <memory>

#include "Eigen/Core"
#include "util/top_n.h"
#include "ctc_beam_entry.h"
#include "ctc_beam_scorer.h"
#include "ctc_decoder.h"
#include "ctc_loss_util.h"

namespace pytorch {
namespace ctc {

template <typename CTCBeamState = ctc_beam_search::EmptyBeamState,
          typename CTCBeamComparer =
              ctc_beam_search::BeamComparer<CTCBeamState>>
class CTCBeamSearchDecoder : public CTCDecoder {
  // Beam Search
  //
  // Example (GravesTh Fig. 7.5):
  //         a    -
  //  P = [ 0.3  0.7 ]  t = 0
  //      [ 0.4  0.6 ]  t = 1
  //
  // Then P(l = -) = P(--) = 0.7 * 0.6 = 0.42
  //      P(l = a) = P(a-) + P(aa) + P(-a) = 0.3*0.4 + ... = 0.58
  //
  // In this case, Best Path decoding is suboptimal.
  //
  // For Beam Search, we use the following main recurrence relations:
  //
  // Relation 1:
  // ---------------------------------------------------------- Eq. 1
  //      P(l=abcd @ t=7) = P(l=abc  @ t=6) * P(d @ 7)
  //                      + P(l=abcd @ t=6) * (P(d @ 7) + P(- @ 7))
  // where P(l=? @ t=7), ? = a, ab, abc, abcd are all stored and
  // updated recursively in the beam entry.
  //
  // Relation 2:
  // ---------------------------------------------------------- Eq. 2
  //      P(l=abc? @ t=3) = P(l=abc @ t=2) * P(? @ 3)
  // for ? in a, b, d, ..., (not including c or the blank index),
  // and the recurrence starts from the beam entry for P(l=abc @ t=2).
  //
  // For this case, the length of the new sequence equals t+1 (t
  // starts at 0).  This special case can be calculated as:
  //   P(l=abc? @ t=3) = P(a @ 0)*P(b @ 1)*P(c @ 2)*P(? @ 3)
  // but we calculate it recursively for speed purposes.
  typedef ctc_beam_search::BeamEntry<CTCBeamState> BeamEntry;
  typedef ctc_beam_search::BeamProbability BeamProbability;

 public:
  typedef BaseBeamScorer<CTCBeamState> DefaultBeamScorer;

  // The beam search decoder is constructed specifying the beam_width (number of
  // candidates to keep at each decoding timestep) and a beam scorer (used for
  // custom scoring, for example enabling the use of a language model).
  // The ownership of the scorer remains with the caller. The default
  // implementation, CTCBeamSearchDecoder<>::DefaultBeamScorer, generates the
  // standard beam search.
  CTCBeamSearchDecoder(int num_classes, int beam_width,
                       BaseBeamScorer<CTCBeamState>* scorer,
                       int blank_index = 0)
      : CTCDecoder(num_classes, blank_index),
        beam_width_(beam_width),
        leaves_(beam_width),
        // TODO: ADD CHECK_NOTNULL BACK
        //beam_scorer_(CHECK_NOTNULL(scorer)) {
        beam_scorer_(scorer) {
    Reset();
  }

  ~CTCBeamSearchDecoder() override {}

  // Run the hibernating beam search algorithm on the given input.
  Status Decode(const CTCDecoder::SequenceLength& seq_len,
                std::vector<CTCDecoder::Input>& input,
                std::vector<CTCDecoder::Output>* output,
                CTCDecoder::ScoreOutput* scores,
                std::vector<CTCDecoder::Output>* alignment,
                std::vector<CTCDecoder::CharProbability>* char_probs) override;

  // Calculate the next step of the beam search and update the internal state.
  template <typename Vector>
  void Step(const Vector& log_input_t, int time_step);

  // Retrieve the beam scorer instance used during decoding.
  BaseBeamScorer<CTCBeamState>* GetBeamScorer() const { return beam_scorer_; }

  // Set label selection parameters for faster decoding.
  // See comments for label_selection_size_ and label_selection_margin_.
  void SetLabelSelectionParameters(int label_selection_size,
                                   float label_selection_margin) {
    label_selection_size_ = label_selection_size;
    label_selection_margin_ = label_selection_margin;
  }

  int GetBeamWidth() const { return beam_width_; }

  // Reset the beam search
  void Reset();

  // Extract the top n paths at current time step
  Status TopPaths(unsigned long n,
    std::vector<std::vector<int>>* paths,
    std::vector<float>* beam_probs,
    std::vector<std::vector<int>>* alignments,
    std::vector<std::vector<float>>* char_probs) const;

 private:
  int beam_width_;

  // Label selection is designed to avoid possibly very expensive scorer calls,
  // by pruning the hypotheses based on the input alone.
  // Label selection size controls how many items in each beam are passed
  // through to the beam scorer. Only items with top N input scores are
  // considered.
  // Label selection margin controls the difference between minimal input score
  // (versus the best scoring label) for an item to be passed to the beam
  // scorer. This margin is expressed in terms of log-probability.
  // Default is to do no label selection.
  // For more detail: https://research.google.com/pubs/pub44823.html
  int label_selection_size_ = 0;       // zero means unlimited
  float label_selection_margin_ = -1;  // -1 means unlimited.

  gtl::TopN<BeamEntry*, CTCBeamComparer> leaves_;
  std::unique_ptr<BeamEntry> beam_root_;
  BaseBeamScorer<CTCBeamState>* beam_scorer_;

  TF_DISALLOW_COPY_AND_ASSIGN(CTCBeamSearchDecoder);
};

// Decode takes the provided input and runs the beam decode using the
// configured scorer. Returns output sequences of labels, beam log probability,
// character alignment, and character probabilities.
template <typename CTCBeamState, typename CTCBeamComparer>
Status CTCBeamSearchDecoder<CTCBeamState, CTCBeamComparer>::Decode(
    const CTCDecoder::SequenceLength& seq_len,
    std::vector<CTCDecoder::Input>& input,
    std::vector<CTCDecoder::Output>* output,
    CTCDecoder::ScoreOutput* beam_probs,
    std::vector<CTCDecoder::Output>* alignment,
    std::vector<CTCDecoder::CharProbability>* char_probs) {
  int batch_size_ = (int)input[0].rows();
  // Storage for top paths.
  std::vector<std::vector<int>> beams;
  std::vector<float> beam_log_probabilities;
  std::vector<std::vector<float>> char_log_probabilities;
  std::vector<std::vector<int>> beam_alignments;
  unsigned long top_n = output->size();

  // check data structure shapes
  if (std::any_of(output->begin(), output->end(),
                  [batch_size_](const CTCDecoder::Output& output) -> bool {
                    return output.size() < batch_size_;
  })) {
    return errors::InvalidArgument("output needs to be of size at least (top_n, batch_size).");
  }
  if (beam_probs->rows() < batch_size_ || beam_probs->cols() < top_n) {
    return errors::InvalidArgument("scores needs to be of size at least (batch_size, top_n).");
  }

  // iterate over each utterance in the batch individually
  for (int b = 0; b < batch_size_; ++b) {
    int seq_len_b = seq_len[b];
    Reset();

    // step through each timestep for the given utterance -- pass in log probabilities of output classes
    for (int t = 0; t < seq_len_b; ++t) {
      Step(input[t].row(b), t);
    }

    // O(n * log(n))
    // complete the score calculation for each beam and re-add to the top-n data structure w/ new scores
    std::unique_ptr<std::vector<BeamEntry*>> branches(leaves_.Extract());
    leaves_.Reset();
    for (int i = 0; i < branches->size(); ++i) {
      BeamEntry* entry = (*branches)[i];
      beam_scorer_->ExpandStateEnd(&entry->state);
      entry->newp.total += beam_scorer_->GetStateEndExpansionScore(entry->state);
      leaves_.push(entry);
    }

    // Extract the top-n paths from the top beams
    Status status = TopPaths(top_n, &beams, &beam_log_probabilities,
                             &beam_alignments, &char_log_probabilities);

    // ensure nothing went awry -- return errors if necessary
    if (!status.ok()) {
      return status;
    }
    if (top_n != beam_log_probabilities.size()) {
      return errors::FailedPrecondition("incorrect number of paths generated");
    }
    if (beams.size() != beam_log_probabilities.size()) {
      return errors::FailedPrecondition("mismatch in beam result sizes");
    }

    // write results to pointers provided by caller
    for (int i = 0; i < top_n; ++i) {
      // Copy output to the correct beam + batch
      (*output)[i][b].swap(beams[i]);
      (*beam_probs)(b, i) = beam_log_probabilities[i];
      (*alignment)[i][b].swap(beam_alignments[i]);
      (*char_probs)[i][b].swap(char_log_probabilities[i]);
    }
  }
  return Status::OK();
}

template <typename CTCBeamState, typename CTCBeamComparer>
template <typename Vector>
void CTCBeamSearchDecoder<CTCBeamState, CTCBeamComparer>::Step(const Vector& raw_input, int time_step) {
  Eigen::ArrayXf input = raw_input;

  // Minimum allowed input value for label selection:
  float label_selection_input_min = -std::numeric_limits<float>::infinity();
  if (label_selection_size_ > 0 && label_selection_size_ < input.size()) {
    std::vector<float> input_copy(input.data(), input.data() + input.size());
    std::nth_element(input_copy.begin(),
                     input_copy.begin() + label_selection_size_ - 1,
                     input_copy.end(), [](float a, float b) { return a > b; });
    label_selection_input_min = input_copy[label_selection_size_ - 1];
  }
  if (label_selection_margin_ >= 0) {
    // max element is 0, per normalization above
    label_selection_input_min =
        std::max(label_selection_input_min, -label_selection_margin_);
  }

  if (num_classes_ != input.size()) {
    return;
  }

  // Extract the beams sorted in decreasing new probability
  std::unique_ptr<std::vector<BeamEntry*>> branches(leaves_.Extract());
  leaves_.Reset();

  for (BeamEntry* b : *branches) {
    // P(.. @ t) becomes the new P(.. @ t-1)
    b->oldp = b->newp;
  }

  // iterate over every current beam to update the probability at that time
  for (BeamEntry* b : *branches) {
    if (b->parent != nullptr) {  // if not the root
      if (b->parent->Active()) {
        // If last two sequence characters are identical:
        //   Plabel(l=acc @ t=6) = (Plabel(l=acc @ t=5)
        //                          + Pblank(l=ac @ t=5))
        // else:
        //   Plabel(l=abc @ t=6) = (Plabel(l=abc @ t=5)
        //                          + P(l=ab @ t=5))
        float previous = (b->label == b->parent->label) ? b->parent->oldp.blank
                                                        : b->parent->oldp.total;
        b->newp.label =
            LogSumExp(b->newp.label,
                      beam_scorer_->GetStateExpansionScore(b->state, previous));
        b->time_step = time_step;
      }
      // Plabel(l=abc @ t=6) *= P(c @ 6)
      b->newp.label += input(b->label);
    }
    // Pblank(l=abc @ t=6) = P(l=abc @ t=5) * P(- @ 6)
    b->newp.blank = b->oldp.total + input(blank_index_);
    // P(l=abc @ t=6) = Plabel(l=abc @ t=6) + Pblank(l=abc @ t=6)
    b->newp.total = LogSumExp(b->newp.blank, b->newp.label);

    // Push the entry back to the top paths list.
    // Note, this will always fill leaves back up in sorted order.
    leaves_.push(b);
  }

  // we need to resort branches in descending oldp order.

  // branches is in descending oldp order because it was
  // originally in descending newp order and we copied newp to oldp.

  // Grow new leaves
  for (BeamEntry* b : *branches) {
    // A new leaf (represented by its BeamProbability) is a candidate
    // iff its total probability is nonzero and either the beam list
    // isn't full, or the lowest probability entry in the beam has a
    // lower probability than the leaf.
    auto is_candidate = [this](const BeamProbability& prob) {
      return (prob.total > kLogZero &&
              (leaves_.size() < beam_width_ ||
               prob.total > leaves_.peek_bottom()->newp.total));
    };

    if (!is_candidate(b->oldp)) {
      continue;
    }

    if (!b->HasChildren()) {
      b->PopulateChildren(num_classes_, blank_index_);
    }

    for (BeamEntry& c : *b->Children()) {
      if (!c.Active()) {
        // Perform label selection: if input for this label looks very
        // unpromising, never evaluate it with a scorer.
        if (input(c.label) < label_selection_input_min) {
          continue;
        }
        //   Pblank(l=abcd @ t=6) = 0
        c.newp.blank = kLogZero;
        // If new child label is identical to beam label:
        //   Plabel(l=abcc @ t=6) = Pblank(l=abc @ t=5) * P(c @ 6)
        // Otherwise:
        //   Plabel(l=abcd @ t=6) = P(l=abc @ t=5) * P(d @ 6)
        beam_scorer_->ExpandState(b->state, b->label, &c.state, c.label);
        float previous = (c.label == b->label) ? b->oldp.blank : b->oldp.total;
        c.newp.label = input(c.label) +
                       beam_scorer_->GetStateExpansionScore(c.state, previous);
        // P(l=abcd @ t=6) = Plabel(l=abcd @ t=6)
        c.newp.total = c.newp.label;

        if (is_candidate(c.newp)) {
          // Before adding the new node to the beam, check if the beam
          // is already at maximum width.
          if (leaves_.size() == beam_width_) {
            // Bottom is no longer in the beam search.  Reset
            // its probability; signal it's no longer in the beam search.
            BeamEntry* bottom = leaves_.peek_bottom();
            bottom->newp.Reset();
          }
          c.time_step = time_step;
          leaves_.push(&c);
        } else {
          // Deactivate child (signal it's not in the beam)
          c.oldp.Reset();
          c.newp.Reset();
        }
      }  // if (!c.Active()) ...
    }    // for (BeamEntry& c in children...
  }      // for (BeamEntry* b...
}

template <typename CTCBeamState, typename CTCBeamComparer>
void CTCBeamSearchDecoder<CTCBeamState, CTCBeamComparer>::Reset() {
  leaves_.Reset();

  // This beam root, and all of its children, will be in memory until
  // the next reset.
  beam_root_.reset(new BeamEntry(nullptr, -1, num_classes_, blank_index_));
  beam_root_->newp.total = 0.0;  // ln(1)
  beam_root_->newp.blank = 0.0;  // ln(1)

  // Add the root as the initial leaf.
  leaves_.push(beam_root_.get());

  // Call initialize state on the root object.
  beam_scorer_->InitializeState(&beam_root_->state);
}

// TopPaths returns `n` sorted top paths based on the internal decoding state.
//
// It accepts pointers to `paths` (character indexes per path), `beam_probs` (
// the neg log likelihood of the beam), `alignments` (the time offset of each
// character), and the `char_probs` (likelihood of the selected character at
// that timestamp).
//
// It does this by evaluating the pushing all BeamEntry `leaves_` into a TopN
// container and returning the top requested N, sorted according to the templated
// CTCBeamComparer.
//
// The top-n returned leaves_ are then traversed back to their tree root to
// generate the n-best character list.
template <typename CTCBeamState, typename CTCBeamComparer>
    Status CTCBeamSearchDecoder<CTCBeamState, CTCBeamComparer>::TopPaths(
        unsigned long n,
        std::vector<std::vector<int>>* paths,
        std::vector<float>* beam_probs,
        std::vector<std::vector<int>>* alignments,
        std::vector<std::vector<float>>* char_probs)
    const {
  if (paths == nullptr || beam_probs == nullptr) {
    return errors::FailedPrecondition(
      "Internal paths are null"
    );
  } else {
    paths->clear();
    beam_probs->clear();
    char_probs->clear();
    alignments->clear();
  }
  if (n > beam_width_) {
    return errors::InvalidArgument("requested more paths than the beam width.");
  }
  if (n > leaves_.size()) {
    return errors::InvalidArgument(
        "Less leaves in the beam search than requested.");
  }

  gtl::TopN<BeamEntry*, CTCBeamComparer> top_branches(n);

  // O(beam_width_ * log(n)), space complexity is O(n)
  for (auto it = leaves_.unsorted_begin(); it != leaves_.unsorted_end(); ++it) {
    top_branches.push(*it);
  }
  // O(n * log(n))
  std::unique_ptr<std::vector<BeamEntry*>> branches(top_branches.Extract());

  for (int i = 0; i < n; ++i) {
    BeamEntry* e((*branches)[i]);
    paths->push_back(e->LabelSeq());
    beam_probs->push_back(e->newp.total);
    char_probs->push_back(e->CharProbSeq());
    alignments->push_back(e->TimeStepSeq());
  }
  return Status::OK();
}

}  // namespace ctc
}  // namespace pytorch

#endif  // PYTORCH_CONTRIB_CTC_CTC_BEAM_SEARCH_H_
