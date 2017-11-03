/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

#ifndef CTC_TRIENODE_H
#define CTC_TRIENODE_H

#include <functional>
#include <istream>
#include <iostream>
#include <limits>

#ifdef INCLUDE_KENLM
#include "lm/model.hh"
#else
namespace lm {
  typedef int WordIndex;
}
#endif

namespace pytorch {
namespace ctc {

#define WORD_MISS_VALUE 123

class TrieNode {
public:
  TrieNode(int vocab_size_) : vocab_size_(vocab_size_),
                        prefix_count_(0),
                        min_score_word_(0),
                        unigram_score_(WORD_MISS_VALUE),
                        min_unigram_score_(std::numeric_limits<float>::max()) {
      children = new TrieNode*[vocab_size_]();
    }

  ~TrieNode() {
    for (int i = 0; i < vocab_size_; i++) {
      delete children[i];
    }
    delete children;
  }

  void WriteToStream(std::ostream& os) {
    WriteNode(os);
    for (int i = 0; i < vocab_size_; i++) {
      if (children[i] == nullptr) {
        os << -1 << std::endl;
      } else {
        // Recursive call
        children[i]->WriteToStream(os);
      }
    }
  }

  static void ReadFromStream(std::istream& is, TrieNode* &obj, int vocab_size_) {
    int prefix_count_;
    is >> prefix_count_;

    if (prefix_count_ == -1) {
      // This is an undefined child
      obj = nullptr;
      return;
    }

    obj = new TrieNode(vocab_size_);
    obj->ReadNode(is, prefix_count_);
    for (int i = 0; i < vocab_size_; i++) {
      // Recursive call
      ReadFromStream(is, obj->children[i], vocab_size_);
    }
  }

  void Insert(const wchar_t* word,
              std::function<int (wchar_t)> translator,
              int lm_word,
              float unigram_score) {
    wchar_t wordCharacter = *word;
    prefix_count_++;
    if (unigram_score_ < min_unigram_score_) {
      min_unigram_score_ = unigram_score;
      min_score_word_ = lm_word;
    }
    if (wordCharacter != '\0') {
      int vocabIndex = translator(wordCharacter);
      TrieNode *child = children[vocabIndex];
      if (child == nullptr)
        child = children[vocabIndex] = new TrieNode(vocab_size_);
      child->Insert(word + 1, translator, lm_word, unigram_score);
    } else {
      unigram_score_ = unigram_score;
    }
  }

  int GetFrequency() {
    return prefix_count_;
  }

  int GetMinScoreWordIndex() {
    return min_score_word_;
  }

  float GetMinUnigramScore() {
    return min_unigram_score_;
  }

  TrieNode *GetChildAt(int vocabIndex) {
    return children[vocabIndex];
  }

  bool GetIsWord() {
    return unigram_score_ != WORD_MISS_VALUE;
  }

  float GetUnigramScore() {
    return unigram_score_;
  }

private:
  int vocab_size_;
  int prefix_count_;
  int min_score_word_;
  float unigram_score_;
  float min_unigram_score_;
  TrieNode **children;

  void WriteNode(std::ostream& os) const {
    os << prefix_count_ << std::endl;
    os << min_score_word_ << std::endl;
    os << min_unigram_score_ << std::endl;
    os << unigram_score_ << std::endl;
  }

  void ReadNode(std::istream& is, int first_input) {
    prefix_count_ = first_input;
    is >> min_score_word_;
    is >> min_unigram_score_;
    is >> unigram_score_;
  }

};

} // namespace ctc
} // namespace pytorch

#endif //CTC_TRIENODE_H
