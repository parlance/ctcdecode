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
    float ngram_score;
    std::wstring word_prefix;
    TrieNode *trie_node;
  };
}

  using pytorch::ctc::ctc_beam_search::DictBeamState;

  class DictBeamScorer : public BaseBeamScorer<DictBeamState> {
   public:

    virtual ~DictBeamScorer() {
      delete trie_root_;
    }

    DictBeamScorer(Labels *labels, const char *trie_path) {
      this->labels_ = labels;

      std::ifstream in;
      in.open(trie_path, std::ios::in);
      TrieNode::ReadFromStream(in, trie_root_, labels_->GetSize());
      in.close();
    }

    // State initialization.
    void InitializeState(DictBeamState* root) const {
      std::cout << "InitializeState" << std::endl;
      root->word_prefix.clear();
      root->trie_node = trie_root_;
      root->ngram_score = 0.0f;
    }

    // void CopyState(const KenLMBeamState& from, KenLMBeamState* to) const {
    //   to->ngram_score = from.ngram_score;
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
      (void)from_label; // unused
      std::cout << "ExpandState" << std::endl;
      //CopyState(from_state, to_state);
      (*to_state) = from_state;

      if (labels_->IsSpace(to_label)) {
        std::cout << "In  IsSpace block" << std::endl;
        //std::wcout << "Old: " << from_state.word_prefix << "; freq=" << from_state.trie_node.GetFrequency() << "; minScoreWordIndex=" << from_state.trie_node.GetMinScoreWordIndex() << "; minUnigramScore" << from_state.trie_node.GetMinUnigramScore() << std::endl;
        //std::wcout << "New: " << to_state->word_prefix << "; freq=" << to_state->trie_node.GetFrequency() << "; minScoreWordIndex=" << to_state->trie_node.GetMinScoreWordIndex() << "; minUnigramScore" << to_state->trie_node.GetMinUnigramScore() << std::endl;
        std::wcout << "Old: " << from_state.word_prefix << "; freq=" << from_state.trie_node->GetFrequency() << "; minScoreWordIndex=" << from_state.trie_node->GetMinScoreWordIndex() << "; minUnigramScore=" << from_state.trie_node->GetMinUnigramScore() << "; score=" << from_state.ngram_score << std::endl;

        to_state->ngram_score = (from_state.trie_node != nullptr) ? 0 : -1000;
        std::wcout << "New: " << to_state->word_prefix << "; freq=" << to_state->trie_node->GetFrequency() << "; minScoreWordIndex=" << to_state->trie_node->GetMinScoreWordIndex() << "; minUnigramScore=" << to_state->trie_node->GetMinUnigramScore() << "; score=" << to_state->ngram_score << std::endl;

        to_state->word_prefix.clear();
        to_state->trie_node = trie_root_;
      } else {
        std::cout << "In !IsSpace block" << std::endl;
        // not at a space, so we don't want to score anything
        to_state->word_prefix += labels_->GetCharacter(to_label);
        if (from_state.trie_node == nullptr) {
          to_state->trie_node = nullptr;
          to_state->ngram_score = -1000;
          std::wcout << "Impossible beam: " << to_state->word_prefix << std::endl;
        } else {

          to_state->trie_node = from_state.trie_node->GetChildAt(to_label);
          to_state->ngram_score = (to_state->trie_node != nullptr) ? 0 : -1000;
          std::wcout << "Possible beam: " << to_state->word_prefix << "; score: " << to_state->ngram_score << std::endl;
        }
      }
    }

    // ExpandStateEnd is called after decoding has finished. Its purpose is to
    // allow a final scoring of the beam in its current state, before resorting
    // and retrieving the TopN requested candidates. Called at most once per beam.
    void ExpandStateEnd(DictBeamState* state) const {
      std::wcout << "ExpandStateEnd: " << state->word_prefix << std::endl;
      state->word_prefix.clear();
      state->trie_node = trie_root_;
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
                                   std::cout << "GetStateExpansionScore" << std::endl;
      return previous_score + state.ngram_score;
    }
    // GetStateEndExpansionScore should be an inexpensive method to retrieve the
    // (cached) expansion score computed within ExpandStateEnd. The score is
    // multiplied (log-addition) with the final probability of the beam.
    //
    // The score returned should be a log-probability.
    float GetStateEndExpansionScore(const DictBeamState& state) const {
      //std::cout << "GetStateEndExpansionScore" << std::endl;
        std::cout << "GetStateEndExpansionScore" << std::endl;
      return state.ngram_score;
    }

   private:
    Labels *labels_;
    TrieNode *trie_root_;
  };

}  // namespace ctc
}  // namespace pytorch

#endif  // PYTORCH_CONTRIB_CTC_CTC_BEAM_SCORER_DICT_H_
