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

#ifndef PYTORCH_CONTRIB_CTC_CTC_BEAM_ENTRY_H_
#define PYTORCH_CONTRIB_CTC_CTC_BEAM_ENTRY_H_

#include <algorithm>
#include <vector>

#include "Eigen/Core"
//#include "tensorflow/core/platform/logging.h"
//#include "tensorflow/core/platform/macros.h"
//#include "tensorflow/core/platform/types.h"
#include "ctc_loss_util.h"
#include "util/macros.h"

// The ctc_beam_search namespace holds several classes meant to be accessed only
// in case of extending the CTCBeamSearch decoder to allow custom scoring
// functions.
//
// BeamEntry is exposed through template arguments BeamScorer and BeamComparer
// of CTCBeamSearch (ctc_beam_search.h).
namespace pytorch {
namespace ctc {
namespace ctc_beam_search {

struct EmptyBeamState {};

struct BeamProbability {
  BeamProbability() : total(kLogZero), blank(kLogZero), label(kLogZero) {}
  void Reset() {
    total = kLogZero;
    blank = kLogZero;
    label = kLogZero;
  }
  float total;
  float blank;
  float label;
};

template <class CTCBeamState = EmptyBeamState>
struct BeamEntry {
  // Default constructor does not create a vector of children.
  BeamEntry() : parent(nullptr), label(-1), time_step(-1) {}
  // Constructor giving parent, label, and number of children does
  // create a vector of children.  The object pointed to by p
  // cannot be copied and should not be moved, otherwise parent will
  // become invalid.
  BeamEntry(BeamEntry* p, int l, int L, int blank)
      : parent(p), label(l), time_step(-1) {
    PopulateChildren(L, blank);
  }
  inline bool Active() const { return newp.total != kLogZero; }
  inline bool HasChildren() const { return !children.empty(); }
  void PopulateChildren(int L, int blank) {
    if (HasChildren()) {
      return;
    }
    children = std::vector<BeamEntry>(L-1);
    for (int ci=0,cl=0; ci < L; ++ci) {
      if (ci == blank) {
        continue;
      }
      // The current object cannot be copied, and should not be moved.
      // Otherwise the child's parent will become invalid.
      auto& c = children[cl];
      c.parent = this;
      c.label = ci;
      ++cl;
    }
  }
  inline std::vector<BeamEntry>* Children() {
    if (!HasChildren()) {
      return {};
    }
    return &children;
  }
  inline const std::vector<BeamEntry>* Children() const {
    if (!HasChildren()) {
      return {};
    }
    return &children;
  }
  std::vector<int> LabelSeq() const {
    std::vector<int> labels;
    const BeamEntry* c = this;
    while (c->parent != nullptr) {  // Checking c->parent to skip root leaf.
      labels.push_back(c->label);
      c = c->parent;
    }
    std::reverse(labels.begin(), labels.end());
    return labels;
  }

  std::vector<int> TimeStepSeq() const {
    std::vector<int> time_steps;
    const BeamEntry *c = this;
    while (c->parent != nullptr) {  // Checking c->parent to skip root leaf.
      time_steps.push_back(c->time_step);
      c = c->parent;
    }
    std::reverse(time_steps.begin(), time_steps.end());
    return time_steps;
  }

  std::vector<float> CharProbSeq() const {
    std::vector<float> probs;
    const BeamEntry *c = this;
    while (c->parent != nullptr) {  // Checking c->parent to skip root leaf.
      probs.push_back(c->newp.total);
      c = c->parent;
    }
    std::reverse(probs.begin(), probs.end());
    return probs;
  }

  BeamEntry<CTCBeamState>* parent;
  int label;
  int time_step;
  std::vector<BeamEntry<CTCBeamState> > children;
  BeamProbability oldp;
  BeamProbability newp;
  CTCBeamState state;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(BeamEntry);
};

// BeamComparer is the default beam comparer provided in CTCBeamSearch.
template <class CTCBeamState = EmptyBeamState>
class BeamComparer {
 public:
  virtual ~BeamComparer() {}
  virtual bool inline operator()(const BeamEntry<CTCBeamState>* a,
                                 const BeamEntry<CTCBeamState>* b) const {
    return a->newp.total > b->newp.total;
  }
};

}  // namespace ctc_beam_search
}
}

#endif  // PYTORCH_CONTRIB_CTC_CTC_BEAM_ENTRY_H_
