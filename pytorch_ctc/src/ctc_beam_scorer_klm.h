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

#ifndef PYTORCH_CONTRIB_CTC_CTC_BEAM_SCORER_KLM_H_
#define PYTORCH_CONTRIB_CTC_CTC_BEAM_SCORER_KLM_H_

#include "ctc_beam_entry.h"
#include "ctc_labels.h"
#include "ctc_trie_node.h"
#include "lm/model.hh"
#include "utf8.h"

#include <iostream>
#include <fstream>

namespace pytorch {
namespace ctc {
namespace ctc_beam_search {
  struct KenLMBeamState {
    float ngram_score;
    int num_words;
    std::wstring word_prefix;
    TrieNode *trie_node;
    lm::ngram::State ngram_state;
  };
}

  using pytorch::ctc::ctc_beam_search::KenLMBeamState;

  class KenLMBeamScorer : public BaseBeamScorer<KenLMBeamState> {
   public:

    ~KenLMBeamScorer() {
      delete model_;
      delete trie_root_;
      delete labels_;
    }
    KenLMBeamScorer(Labels *labels, const char *kenlm_path, const char *trie_path)
                      : ngram_model_weight_(1.0f),
                        word_insertion_weight_(1.0f),
                        default_min_unigram_(kLogZero) {
      lm::ngram::Config config;
      config.load_method = util::POPULATE_OR_READ;
      model_ = lm::ngram::LoadVirtual(kenlm_path, config);
      this->labels_ = labels;

      std::ifstream in;
      in.open(trie_path, std::ios::in);
      TrieNode::ReadFromStream(in, trie_root_, labels_->GetSize());
      in.close();
    }

    // State initialization.
    void InitializeState(KenLMBeamState* root) const {
      //std::cout << "InitializeState" << std::endl;
      root->ngram_score = 0.0f;
      root->num_words = 0;
      root->word_prefix.clear();
      //model->BeginSentenceWrite(&root->ngram_state);
      model_->NullContextWrite(&root->ngram_state);
    }

    // ExpandState is called when expanding a beam to one of its children.
    // Called at most once per child beam. In the simplest case, no state
    // expansion is done.
    void ExpandState(const KenLMBeamState& from_state, int from_label,
                           KenLMBeamState* to_state, int to_label) const {
      (void)from_label; // unused
      //CopyState(from_state, to_state);
      (*to_state) = from_state;

      if (labels_->IsSpace(to_label)) {
        // if there are words already (ie this isnt just whitespace) increment word count
        if (!from_state.word_prefix.empty()) {
          ++to_state->num_words;
        }

        if (IsWord(from_state.word_prefix)) {

          //std::wcout << "trying to score: " << from_state.word_prefix << "; num_words: " << to_state->num_words << std::endl;
          to_state->ngram_score = ScoreNewWord(from_state.ngram_state, from_state.word_prefix,
                                               &to_state->ngram_state);
        //  for (int i=0; i < (int)to_state->ngram_state.Length(); ++i) {
        //    std::cout << "    " << to_state->ngram_state.words[i] << std::endl;
        //  }
        } else {
          to_state->ngram_score = default_min_unigram_;
        }
      } else {
        // not at a space, so we don't want to score anything
        to_state->word_prefix += labels_->GetCharacter(to_label);
        to_state->ngram_state = from_state.ngram_state;
        if(IsPrefix(to_state->word_prefix)){
          to_state->ngram_score = 0;
        } else {
          to_state->ngram_score = default_min_unigram_;
        }
      }
    }

    // ExpandStateEnd is called after decoding has finished. Its purpose is to
    // allow a final scoring of the beam in its current state, before resorting
    // and retrieving the TopN requested candidates. Called at most once per beam.
    void ExpandStateEnd(KenLMBeamState* state) const {
      if (!state->word_prefix.empty()) {
        if (IsWord(state->word_prefix)){
          ++state->num_words;
          lm::ngram::State to_ngram_state;
          state->ngram_score = ScoreNewWord(state->ngram_state, state->word_prefix, &to_ngram_state);
        } else {
          state->ngram_score = default_min_unigram_;
        }
      } else {
        state->ngram_score = 0;
      }
    }

    // GetStateExpansionScore should be an inexpensive method to retrieve the
    // (cached) expansion score computed within ExpandState. The score is
    // multiplied (log-addition) with the input score at the current step from
    // the network.
    //
    // The score returned should be a log-probability. In the simplest case, as
    // there's no state expansion logic, the expansion score is zero.
    float GetStateExpansionScore(const KenLMBeamState& state,
                                 float previous_score) const {
      return previous_score + ngram_model_weight_*state.ngram_score +
             word_insertion_weight_ * state.num_words;
    }
    // GetStateEndExpansionScore should be an inexpensive method to retrieve the
    // (cached) expansion score computed within ExpandStateEnd. The score is
    // multiplied (log-addition) with the final probability of the beam.
    //
    // The score returned should be a log-probability.
    float GetStateEndExpansionScore(const KenLMBeamState& state) const {
      //std::cout << "GetStateEndExpansionScore" << std::endl;
      return ngram_model_weight_ * state.ngram_score +
             word_insertion_weight_ * state.num_words;
    }

    void SetLMWeight(float lm_weight) {
      this->ngram_model_weight_ = lm_weight;
    }

    void SetWordCountWeight(float word_count_weight) {
      this->word_insertion_weight_ = word_count_weight;
    }

    void SetValidWordCountWeight(float valid_word_count_weight) {

    }

   private:
    Labels *labels_;
    TrieNode *trie_root_;
    lm::base::Model *model_;
    float ngram_model_weight_;
    float word_insertion_weight_;
    float default_min_unigram_;

    float ScoreNewWord(const lm::ngram::State& from_ngram_state,
                                    const std::wstring& new_word,
                                    lm::ngram::State* to_ngram_state) const {

        std::string encoded_word;
        utf8::utf16to8(new_word.begin(), new_word.end(), std::back_inserter(encoded_word));
        return model_->BaseScore(&from_ngram_state,
                                 model_->BaseVocabulary().Index(encoded_word),
                                 to_ngram_state);
    }

    bool IsPrefix(const std::wstring& prefix) const {
      std::wcout << "Checking IsPrefix on: " << prefix << std::endl;
      TrieNode *tn = trie_root_;
      for (const wchar_t c : prefix) {
        if (c == '\0') {
          break;
        }
        int to_label = labels_->GetLabel(c);
        tn = tn->GetChildAt(to_label);
        if (tn == nullptr) {
          std::wcout << "Checking IsPrefix on: " << prefix << " returns FALSE" << std::endl;
          return false;
        }
      }
      std::wcout << "Checking IsPrefix on: " << prefix << " returns TRUE" << std::endl;
      return true;
    }

    bool IsWord(const std::wstring& word) const {
      std::string encoded_word;
      utf8::utf16to8(word.begin(), word.end(), std::back_inserter(encoded_word));
      auto &vocabulary = model_->BaseVocabulary();
      return vocabulary.Index(encoded_word) != vocabulary.NotFound();
    }
  };

}  // namespace ctc
}  // namespace pytorch

#endif  // PYTORCH_CONTRIB_CTC_CTC_BEAM_SCORER_KLM_H_
