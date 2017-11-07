#ifndef PYTORCH_CONTRIB_CTC_CTC_BEAM_SCORER_DICT_H_
#define PYTORCH_CONTRIB_CTC_CTC_BEAM_SCORER_DICT_H_

#include "ctc_beam_entry.h"
#include "ctc_labels.h"
#include "ctc_trie_node.h"
#include "utf8.h"

#include <iostream>
#include <fstream>

namespace pytorch {
namespace ctc {
namespace ctc_beam_search {
  struct DictBeamState {
    float score;
    TrieNode *node;
  };
}

  using pytorch::ctc::ctc_beam_search::DictBeamState;

  class DictBeamScorer : public BaseBeamScorer<DictBeamState> {
   public:

    virtual ~DictBeamScorer() {
      delete trie_root_;
    }

    DictBeamScorer(Labels *labels, const char *trie_path) :
        labels_(labels),
        default_min_unigram_(kLogZero)
    {
      std::ifstream in;
      in.open(trie_path, std::ios::in);
      TrieNode::ReadFromStream(in, trie_root_, labels_->GetSize());
      in.close();
    }

    // State initialization.
    void InitializeState(DictBeamState* root) const {
      root->node = trie_root_;
      root->score = 0.0f;
    }

    // ExpandState is called when expanding a beam to one of its children.
    // Called at most once per child beam. In the simplest case, no state
    // expansion is done.
    void ExpandState(const DictBeamState& from_state, int from_label,
                           DictBeamState* to_state, int to_label) const {
      // check to see if we're on a word boundary
      if (labels_->IsSpace(to_label) && from_state.node != trie_root_) {
        // check if from_state is valid
        to_state->score = StateIsCandidate(from_state, true) ? 0.0 : default_min_unigram_;
        to_state->node = trie_root_;
      } else {
        to_state->node = (from_state.node == nullptr) ? nullptr : from_state.node->GetChildAt(to_label);
        to_state->score = StateIsCandidate(*to_state, false) ? 0.0 : default_min_unigram_;
      }
    }


    // ExpandStateEnd is called after decoding has finished. Its purpose is to
    // allow a final scoring of the beam in its current state, before resorting
    // and retrieving the TopN requested candidates. Called at most once per beam.
    void ExpandStateEnd(DictBeamState* state) const {
      state->score = StateIsCandidate(*state, true) ? 0.0 : default_min_unigram_;
      state->node = trie_root_;
    }

    // GetStateExpansionScore should be an inexpensive method to retrieve the
    // (cached) expansion score computed within ExpandState. The score is
    // multiplied (log-addition) with the input score at the current step from
    // the network.
    //
    // The score returned should be a log-probability. In the simplest case, as
    // there's no state expansion logic, the expansion score is zero.
    float GetStateExpansionScore(const DictBeamState& state,
                                 float previous_score) const {
                                   //std::cout << "GetStateExpansionScore" << std::endl;
       return previous_score + state.score;
    }
    // GetStateEndExpansionScore should be an inexpensive method to retrieve the
    // (cached) expansion score computed within ExpandStateEnd. The score is
    // multiplied (log-addition) with the final probability of the beam.
    //
    // The score returned should be a log-probability.
    float GetStateEndExpansionScore(const DictBeamState& state) const {
        return state.score;
    }

    void SetMinimumUnigramProbability(float min_unigram) {
      this->default_min_unigram_ = min_unigram;
    }

   private:
    Labels *labels_;
    TrieNode *trie_root_;
    float default_min_unigram_;

    bool StateIsCandidate(const DictBeamState& state, bool word) const;
  };

  bool DictBeamScorer::StateIsCandidate(
    const DictBeamState& state, bool word) const {
    // Check if the beam can still be a dictionary word (e.g. prefix of one).
    if ((state.node == nullptr) || (word && !state.node->GetIsWord())) {
      return false;
    }
    return true;
  }

}  // namespace ctc
}  // namespace pytorch

#endif  // PYTORCH_CONTRIB_CTC_CTC_BEAM_SCORER_DICT_H_
