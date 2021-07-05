#include "path_trie.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <utility>
#include <vector>
#include <map>

#include "decoder_utils.h"

PathTrie::PathTrie() {
  log_prob_b_prev = -NUM_FLT_INF;
  log_prob_nb_prev = -NUM_FLT_INF;
  log_prob_b_cur = -NUM_FLT_INF;
  log_prob_nb_cur = -NUM_FLT_INF;
  log_prob_c = -NUM_FLT_INF;
  score = -NUM_FLT_INF;

  ROOT_ = -1;
  character = ROOT_;
  timestep = 0;
  exists_ = true;
  parent = nullptr;

  dictionary_ = nullptr;
  dictionary_state_ = 0;
  has_dictionary_ = false;

  matcher_ = nullptr;

  has_mini_dictionary_ = false;
  mini_dictionary_ = nullptr;
  mini_dictionary_state_ = 0;
  mini_matcher_ = nullptr;


  vocab_tmp = {
    "_", "ー", "あ", "い", "う", "え", "お", "か", "き", "く", "け", "こ", "さ", "し", "す", "せ", "そ", "た", "ち", "つ", "て", "と", "な", "に", "ぬ", "ね", "の", "は", "ひ", "ふ", "へ", "ほ", "ま", "み", "む", "め", "も", "や", "ゆ", "よ", "ら", "り", "る", "れ", "ろ", "わ", "を", "ん", "ぁ", "ぃ", "ぅ", "ぇ", "ぉ", "ゃ", "ゅ", "ょ", "ゎ", "っ", "が", "ぎ", "ぐ", "げ", "ご", "ざ", "じ", "ず", "ぜ", "ぞ", "だ", "ぢ", "づ", "で", "ど", "ば", "び", "ぶ", "べ", "ぼ", "ぱ", "ぴ", "ぷ", "ぺ", "ぽ", "ゔ", " "
  };
}

PathTrie::~PathTrie() {
  for (auto child : children_) {
    delete child.second;
  }
}

// actually create and keep adding new trie nodes.
PathTrie* PathTrie::get_path_trie(int new_char, int new_timestep, float cur_log_prob_c, bool reset) {
  auto child = children_.begin();
  for (child = children_.begin(); child != children_.end(); ++child) {
    if (child->first == new_char) {
      if (child->second->log_prob_c < cur_log_prob_c) {
        child->second->log_prob_c = cur_log_prob_c;
        child->second->timestep = new_timestep;
      }
      break;
    }
  }
  if (child != children_.end()) {
    if (!child->second->exists_) {
      child->second->exists_ = true;
      child->second->log_prob_b_prev = -NUM_FLT_INF;
      child->second->log_prob_nb_prev = -NUM_FLT_INF;
      child->second->log_prob_b_cur = -NUM_FLT_INF;
      child->second->log_prob_nb_cur = -NUM_FLT_INF;
    }
    return (child->second);

  } else {

    if (has_dictionary_) {

      if (has_mini_dictionary_) {

        // both base and mini dictinary

        // // show log
        // std::ofstream ofs13334("/tmp/cpp_log.txt", std::ios::app);
        // ofs13334 << "has_dictionary_ has_mini_dictionary_ : " << "  " << std::endl;
        // ofs13334.close();

        // evaluate dictonary and mini-dictonary one by one

        // evaluate with base dictionar
        bool found = false;
        if (base_dictionary_active == true) {
          matcher_->SetState(dictionary_state_);
          found = matcher_->Find(new_char + 1);
        }
        // matcher_->SetState(dictionary_state_);
        // bool found = matcher_->Find(new_char + 1);

        // evaluate with mini dictionary
        bool found_min = false;
        if (mini_dictionary_active == true) {

          mini_matcher_->SetState(mini_dictionary_state_);
          found_min = mini_matcher_->Find(new_char + 1);
        }

        if (!found) {

          // Adding this character causes word outside dictionary
          auto FSTZERO = fst::TropicalWeight::Zero();
          auto final_weight = dictionary_->Final(dictionary_state_);
          bool is_final = (final_weight != FSTZERO);
          if (is_final && reset) {

            // // WTF?! Nerver come here...
            // // show log
            // std::ofstream ofs13334("/tmp/cpp_log.txt", std::ios::app);
            // ofs13334 << "dictionary_reset : " << "  " << std::endl;
            // ofs13334.close();

            dictionary_state_ = dictionary_->Start();
          }
        }

        if (!found_min) {
          // Adding this character causes word outside dictionary
          auto FSTZERO = fst::TropicalWeight::Zero();
          auto final_weight = mini_dictionary_->Final(mini_dictionary_state_);
          bool is_final = (final_weight != FSTZERO);
          if (is_final && reset) {

            // // WTF?! Nerver come here...
            // // show log
            // std::ofstream ofs13335("/tmp/cpp_log.txt", std::ios::app);
            // ofs13335 << "mini_dictionary_reset : " << "  " << std::endl;
            // ofs13335.close();

            mini_dictionary_state_ = mini_dictionary_->Start();

          }
        }

        // if (found_min) {
        //
        //   // show log
        //   std::ofstream ofs13334("/tmp/cpp_log.txt", std::ios::app);
        //   ofs13334 << "found_min TRUE : " << "  " << std::endl;
        //   ofs13334.close();
        // }

        // not found at all both in base and mini dictionary
        // ここでnullptrを適切に返せないとぐちゃぐちゃと繋がった単語が出る。
        if (!found && !found_min) {

          return nullptr;

        } else {

          PathTrie* new_path = new PathTrie;
          new_path->character = new_char;
          new_path->timestep = new_timestep;
          new_path->parent = this;
          new_path->dictionary_ = dictionary_;
          new_path->has_dictionary_ = true;
          new_path->matcher_ = matcher_;
          new_path->log_prob_c = cur_log_prob_c;
          // funnels
          new_path->mini_dictionary_ = mini_dictionary_;
          new_path->has_mini_dictionary_ = true;
          new_path->mini_matcher_ = mini_matcher_;

          if (found) {
            // set spell checker state
            // check to see if next state is final
            auto FSTZERO = fst::TropicalWeight::Zero();
            auto final_weight = dictionary_->Final(matcher_->Value().nextstate);
            bool is_final = (final_weight != FSTZERO);
            if (is_final && reset) {

              // // show log
              // std::ofstream ofs13345("/tmp/cpp_log.txt", std::ios::app);
              // ofs13345 << "3.mini_dictionary_reset : " << vocab_tmp[new_char] << std::endl;
              // ofs13345.close();

              // restart spell checker at the start state
              new_path->dictionary_state_ = dictionary_->Start();

              // new_path->base_dictionary_active = false;

            } else {


              // // show log
              // std::ofstream ofs13355("/tmp/cpp_log.txt", std::ios::app);
              // ofs13355 << "3.mini_dictionary_continue : " << vocab_tmp[new_char] << std::endl;
              // ofs13355.close();

              // new_path->base_dictionary_active = true;

               // go to next state
              new_path->dictionary_state_ = matcher_->Value().nextstate;
            }


            new_path->base_dictionary_active = true;


          } else {

            // // show log
            // std::ofstream ofs13355("/tmp/cpp_log.txt", std::ios::app);
            // ofs13355 << "1. kokoha?: " << std::endl;
            // ofs13355.close();
            // new_path->dictionary_state_ = dictionary_->Start();
            new_path->base_dictionary_active = false;
          }

          if (found_min) {
            auto FSTZERO = fst::TropicalWeight::Zero();
            auto final_weight_mini = mini_dictionary_->Final(mini_matcher_->Value().nextstate);
            bool is_final_mini = (final_weight_mini != FSTZERO);
            if (is_final_mini && reset) {

              //
              // // show log
              // std::ofstream ofs13346("/tmp/cpp_log.txt", std::ios::app);
              // ofs13346 << "4.mini_dictionary_reset : " << vocab_tmp[new_char] << std::endl;
              // ofs13346.close();

              // restart spell checker at the start state
              new_path->mini_dictionary_state_ = mini_dictionary_->Start();

              // new_path->mini_dictionary_active = false;

            } else {

              // // show log
              // std::ofstream ofs13365("/tmp/cpp_log.txt", std::ios::app);
              // ofs13365 << "4.mini_dictionary_continue : " << vocab_tmp[new_char] << std::endl;
              // ofs13365.close();

              // new_path->mini_dictionary_active = true;

              // go to next state
              new_path->mini_dictionary_state_ = mini_matcher_->Value().nextstate;
            }

            new_path->mini_dictionary_active = true;

          } else {

            // // show log
            // std::ofstream ofs13365("/tmp/cpp_log.txt", std::ios::app);
            // ofs13365 << "2.kokoha : " << std::endl;
            // ofs13365.close();

            // unfortunately if we set this false, we could not get funnel words!

            // new_path->mini_dictionary_state_ = mini_dictionary_->Start();
            new_path->mini_dictionary_active = false;

            // new_path->mini_dictionary_active = true;

            // if (base_dictionary_active == true) {
            //   new_path->mini_dictionary_active = true;
            // } else {
            //   // new_path->mini_dictionary_state_ = mini_dictionary_->Start();
            //   new_path->mini_dictionary_active = false;
            // }
          }

          // new trie node and return
          children_.push_back(std::make_pair(new_char, new_path));
          return new_path;
        }

      } else {

        // base dictionar only

        matcher_->SetState(dictionary_state_);
        bool found = matcher_->Find(new_char + 1);
        if (!found) {
          // Adding this character causes word outside dictionary
          auto FSTZERO = fst::TropicalWeight::Zero();
          auto final_weight = dictionary_->Final(dictionary_state_);
          bool is_final = (final_weight != FSTZERO);
          if (is_final && reset) {

            // // WTF?! Nerver come here...
            // // show log
            // std::ofstream ofs13335("/tmp/cpp_log.txt", std::ios::app);
            // ofs13335 << "1.mini_dictionary_reset : " << "  " << std::endl;
            // ofs13335.close();

            dictionary_state_ = dictionary_->Start();
          }
          return nullptr;

        } else {

          PathTrie* new_path = new PathTrie;
          new_path->character = new_char;
          new_path->timestep = new_timestep;
          new_path->parent = this;
          new_path->dictionary_ = dictionary_;
          new_path->has_dictionary_ = true;
          new_path->matcher_ = matcher_;
          new_path->log_prob_c = cur_log_prob_c;

          // set spell checker state
          // check to see if next state is final
          auto FSTZERO = fst::TropicalWeight::Zero();
          auto final_weight = dictionary_->Final(matcher_->Value().nextstate);
          bool is_final = (final_weight != FSTZERO);
          if (is_final && reset) {
  	         // restart spell checker at the start state

             // // show log
             // std::ofstream ofs13335("/tmp/cpp_log.txt", std::ios::app);
             // ofs13335 << "2.mini_dictionary_reset : " << "  " << std::endl;
             // ofs13335.close();

            new_path->dictionary_state_ = dictionary_->Start();

          } else {
  	         // go to next state
            new_path->dictionary_state_ = matcher_->Value().nextstate;
          }

          children_.push_back(std::make_pair(new_char, new_path));
          return new_path;
        }

      }

    } else {
      PathTrie* new_path = new PathTrie;
      new_path->character = new_char;
      new_path->timestep = new_timestep;
      new_path->parent = this;
      new_path->log_prob_c = cur_log_prob_c;
      children_.push_back(std::make_pair(new_char, new_path));
      return new_path;
    }
  }
}

PathTrie* PathTrie::get_path_vec(std::vector<int>& output, std::vector<int>& timesteps) {
  return get_path_vec(output, timesteps, ROOT_);
}

PathTrie* PathTrie::get_path_vec(std::vector<int>& output,
                                 std::vector<int>& timesteps,
                                 int stop,
                                 size_t max_steps) {

  if (character == stop || character == ROOT_ || output.size() == max_steps) {
    std::reverse(output.begin(), output.end());
    std::reverse(timesteps.begin(), timesteps.end());
    return this;
  } else {
    output.push_back(character);
    timesteps.push_back(timestep);
    return parent->get_path_vec(output, timesteps, stop, max_steps);
  }
}

void PathTrie::iterate_to_vec(std::vector<PathTrie*>& output) {

  if (exists_) {
    log_prob_b_prev = log_prob_b_cur;
    log_prob_nb_prev = log_prob_nb_cur;

    log_prob_b_cur = -NUM_FLT_INF;
    log_prob_nb_cur = -NUM_FLT_INF;

    score = log_sum_exp(log_prob_b_prev, log_prob_nb_prev);

    output.push_back(this);
  }

  for (auto child : children_) {

    child.second->iterate_to_vec(output);
  }
}

void PathTrie::remove() {
  exists_ = false;

  if (children_.size() == 0) {
    auto child = parent->children_.begin();
    for (child = parent->children_.begin(); child != parent->children_.end();
         ++child) {
      if (child->first == character) {
        parent->children_.erase(child);
        break;
      }
    }

    if (parent->children_.size() == 0 && !parent->exists_) {
      parent->remove();
    }

    delete this;
  }
}

void PathTrie::set_dictionary(fst::StdVectorFst* dictionary) {
  dictionary_ = dictionary;
  dictionary_state_ = dictionary->Start();
  has_dictionary_ = true;
}

using FSTMATCH = fst::SortedMatcher<fst::StdVectorFst>;
void PathTrie::set_matcher(std::shared_ptr<FSTMATCH> matcher) {
  matcher_ = matcher;
}

/// funnels

// create mini dictionary
// copy from "void Scorer::fill_dictionary(bool add_space);"
double PathTrie::create_mini_dictionary(const std::map<std::string, std::string> &funnels,
                                      std::unordered_map<std::string, int> char_map_,
                                      int space_id,
                                      bool add_space) {

  fst::StdVectorFst dictionary;

  // For each unigram convert to ints and put in trie
  // int dict_size = 0;

  for (const auto& [key, value] : funnels){

    // why we need to add 1 to space_id ?
    bool added = add_word_to_dictionary(
        key, char_map_, add_space, space_id + 1, &dictionary);
  }


  /* Simplify FST

   * This gets rid of "epsilon" transitions in the FST.
   * These are transitions that don't require a string input to be taken.
   * Getting rid of them is necessary to make the FST determinisitc, but
   * can greatly increase the size of the FST
   */
  fst::RmEpsilon(&dictionary);
  fst::StdVectorFst* new_dict = new fst::StdVectorFst;

  /* This makes the FST deterministic, meaning for any string input there's
   * only one possible state the FST could be in.  It is assumed our
   * dictionary is deterministic when using it.
   * (lest we'd have to check for multiple transitions at each state)
   */
  fst::Determinize(dictionary, new_dict);

  /* Finds the simplest equivalent fst. This is unnecessary but decreases
   * memory usage of the dictionary
   */
  fst::Minimize(new_dict);
  // this->dictionary = new_dict;

  mini_dictionary_ = new_dict;
  mini_dictionary_state_ = mini_dictionary_->Start();
  has_mini_dictionary_ = true;

  // setup matcher
  auto mini_matcher = std::make_shared<FSTMATCH>(mini_dictionary_, fst::MATCH_INPUT);
  mini_matcher_ = mini_matcher;
}

// void PathTrie::set_mini_matcher(std::shared_ptr<FSTMATCH> matcher) {
//   mini_matcher_ = matcher;
// }
