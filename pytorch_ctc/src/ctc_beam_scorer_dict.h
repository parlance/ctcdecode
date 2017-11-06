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

// Collection of scoring classes that can be extended and provided to the
// CTCBeamSearchDecoder to incorporate additional scoring logic (such as a
// language model).
//
// To build a custom scorer extend and implement the pure virtual methods from
// BeamScorerInterface. The default CTC decoding behavior is implemented
// through BaseBeamScorer.

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
        labels_(labels)
    {
      std::ifstream in;
      in.open(trie_path, std::ios::in);
      TrieNode::ReadFromStream(in, trie_root_, labels_->GetSize());
      in.close();
    }

    // State initialization.
    void InitializeState(DictBeamState* root) const {
      //std::cout << "InitializeState" << std::endl;
      root->node = trie_root_;
      root->score = 0.0f;
    }

    // void CopyState(const KenLMBeamState& from, KenLMBeamState* to) const {
    //   to->score = from.score;
    //   to->num_words = from.num_words;
    //   to->word_prefix = from.word_prefix;
    //   to->trie_node = from.trie_node;
    //   to->ngram_state = from.ngram_state;
    // }

    // ExpandState is called when expanding a beam to one of its children.
    // Called at most once per child beam. In the simplest case, no state
    // expansion is done.
    void ExpandState(const DictBeamState& from_state, int from_label,
                           DictBeamState* to_state, int to_label) const {
      // check to see if we're on a word boundary
      if (labels_->IsSpace(to_label) && from_state.node != trie_root_) {
        // check if from_state is valid
        to_state->score = StateIsCandidate(from_state, true) ? 0.0 : kLogZero;
        to_state->node = trie_root_;
      } else {
        to_state->node = (from_state.node == nullptr) ? nullptr : from_state.node->GetChildAt(to_label);
        to_state->score = StateIsCandidate(*to_state, false) ? 0.0 : kLogZero;
      }
    }


    // ExpandStateEnd is called after decoding has finished. Its purpose is to
    // allow a final scoring of the beam in its current state, before resorting
    // and retrieving the TopN requested candidates. Called at most once per beam.
    void ExpandStateEnd(DictBeamState* state) const {
      //std::wcout << "ExpandStateEnd: " << state->word_prefix << std::endl;
      //state->word_prefix.clear();
      state->score = StateIsCandidate(*state, true) ? 0.0 : kLogZero;
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
      //std::cout << "GetStateEndExpansionScore" << std::endl;
        //std::cout << "GetStateEndExpansionScore" << std::endl;
        return state.score;
    }

   private:
    Labels *labels_;
    TrieNode *trie_root_;

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
