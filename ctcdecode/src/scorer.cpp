#include "scorer.h"

#include <unistd.h>
#include <iostream>

#include <map>

#include "lm/config.hh"
#include "lm/model.hh"
#include "lm/state.hh"
#include "util/string_piece.hh"
#include "util/tokenize_piece.hh"

#include "decoder_utils.h"

using namespace lm::ngram;

Scorer::Scorer(double alpha,
               double beta,
               const std::string& lm_path,
               const std::vector<std::string>& vocab_list) {
  this->alpha = alpha;
  this->beta = beta;

  dictionary = nullptr;
  is_character_based_ = true;
  language_model_ = nullptr;

  max_order_ = 0;
  dict_size_ = 0;
  SPACE_ID_ = -1;

  setup(lm_path, vocab_list);
}

Scorer::~Scorer() {
  if (language_model_ != nullptr) {
    delete static_cast<lm::base::Model*>(language_model_);
  }
  if (dictionary != nullptr) {
    delete static_cast<fst::StdVectorFst*>(dictionary);
  }
}

void Scorer::setup(const std::string& lm_path,
                   const std::vector<std::string>& vocab_list) {

  // load language model
  load_lm(lm_path);

  // set char map for scorer
  set_char_map(vocab_list);

  // fill the dictionary for FST
  if (!is_character_based()) {
    fill_dictionary(true);
  }
}

void Scorer::load_lm(const std::string& lm_path) {
  const char* filename = lm_path.c_str();
  VALID_CHECK_EQ(access(filename, F_OK), 0, "Invalid language model path");

  RetriveStrEnumerateVocab enumerate;
  lm::ngram::Config config;
  config.enumerate_vocab = &enumerate;
  language_model_ = lm::ngram::LoadVirtual(filename, config);
  max_order_ = static_cast<lm::base::Model*>(language_model_)->Order();
  vocabulary_ = enumerate.vocabulary;

  for (size_t i = 0; i < vocabulary_.size(); ++i) {

    if (is_character_based_ && vocabulary_[i] != UNK_TOKEN &&
        vocabulary_[i] != START_TOKEN && vocabulary_[i] != END_TOKEN &&
        get_utf8_str_len(enumerate.vocabulary[i]) > 1) {
      is_character_based_ = false;
    }
  }
}

// get score with lm
double Scorer::get_log_cond_prob(const std::vector<std::string>& words,
                                 const std::map<std::string, std::string> &funnels,
                                 const std::map<std::string, double> &weights ) {

  lm::base::Model* model = static_cast<lm::base::Model*>(language_model_);

  double cond_prob;

  lm::ngram::State state, tmp_state, out_state;

  // avoid to inserting <s> in begin
  model->NullContextWrite(&state);

  bool found_in_both = 0;
  bool found_in_funnel = 0;

  double weight_on_new = 0.9
  double weight_on_exist = 0.7

  // ////////////
  // // lm::WordIndex word_index4 = model->BaseVocabulary().Index("かくにん");
  // std::ofstream ofs23537("/tmp/cpp_log.txt", std::ios::app);
  // ofs23537 << "funnels.size: " << funnels.size() << funnels[0] << std::endl;
  // ofs23537.close();
  // //////////

  for (size_t i = 0; i < words.size(); ++i) {

    // if ( words[i] == "ねっちゅうしょう" ) {
    //
    //   // show log
    //   std::ofstream ofs23334("/tmp/cpp_log.txt", std::ios::app);
    //   ofs23334 << "word_index === 0: " << words[i] << std::endl;
    //   ofs23334.close();
    // }

    lm::WordIndex word_index = model->BaseVocabulary().Index(words[i]);

    // try funnels
    if (word_index == 0) {

      // search words
      if (funnels.size() > 0) {

        // if ( auto iter = funnels.find(words[i]); iter != end(funnels) ) {
        if ( funnels.find(words[i]) != funnels.end() ) {


          // show log
          std::ofstream ofs23334("/tmp/cpp_log.txt", std::ios::app);
          ofs23334 << "words00: " << words[i] << std::endl;
          ofs23334.close();

          found_in_funnel = 1;

          // update weights
          if (weights.size() > 0) {
            if ( weights.find(words[i]) != weights.end() ) {
              weight_on_new = weights.at(words[i])
            }
          }

          if (funnels.at(words[i]) != "default") {

            lm::WordIndex word_index2 = model->BaseVocabulary().Index(funnels.at(words[i]));
            word_index = word_index2;

          } else {
            // std::string score_of(funnels[words[i]]);
            lm::WordIndex word_index2 = model->BaseVocabulary().Index("ぱそこん");
            word_index = word_index2;
          }

        }
      }

    } else {

      if (funnels.size() > 0) {


        // if ( words[i] == "ふぁし" ) {
        //
        //   // show log
        //   std::ofstream ofs23334("/tmp/cpp_log.txt", std::ios::app);
        //   ofs23334 << "word_index !== 0: " << words[i] << std::endl;
        //   ofs23334.close();
        // }


        // if ( auto iter = funnels.find(words[i]); iter != end(funnels) ) {
        if ( funnels.find(words[i]) != funnels.end() ) {

          // show log
          std::ofstream ofs29334("/tmp/cpp_log.txt", std::ios::app);
          ofs29334 << "found_in_both: " << words[i] << std::endl;
          ofs29334.close();

          found_in_both = 1;

          // update weights
          if (weights.size() > 0) {
            if ( weights.find(words[i]) != weights.end() ) {
              weight_on_exist = weights.at(words[i])
            }
          }

        }
      }
    }

    // encounter OOV
    if (word_index == 0) {


      // if ( words[i] == "だこく" ) {
      //
      //   // show log
      //   std::ofstream ofs23334("/tmp/cpp_log.txt", std::ios::app);
      //   ofs23334 << "word_index === 0: " << words[i] << std::endl;
      //   ofs23334.close();
      //
      // }


      //  -1000.0;
      return OOV_SCORE;
    }

    // this one get score from KenLM
    // input current state, new word index, and new state
    cond_prob = model->BaseScore(&state, word_index, &out_state);

    tmp_state = state;
    state = out_state;
    out_state = tmp_state;


    // if (found_in_funnel == true) {
    //
    //   // show log
    //   std::ofstream ofs23337("/tmp/cpp_log.txt", std::ios::app);
    //   ofs23337 << "cond_prob: " << cond_prob << " " << words[i] << std::endl;
    //   ofs23337.close();
    // }
  }

  if (found_in_both == true) {

    // // show sentence
    // string sentence("");
    // for (int i = 0; i < words.size(); i++ ) {
    //   // append
    //   sentence.append(" ");
    //   sentence.append(words[i]);
    // }
    // std::ofstream ofs23338("/tmp/cpp_log.txt", std::ios::app);
    // ofs23338 << "sentence1: " << cond_prob << " " << sentence << std::endl;
    // ofs23338.close();

    // 0.1 is just a number. you can experiment other numbers.
    cond_prob = cond_prob * weight_on_new;

  } else if (found_in_funnel == true) {

    // // show log
    // std::ofstream ofs23337("/tmp/cpp_log.txt", std::ios::app);
    // ofs23337 << "cond_prob2222: " << cond_prob << " " << words[i] << std::endl;
    // ofs23337.close();

    // // show sentence log
    // string sentence("");
    // for (int i = 0; i < words.size(); i++ ) {
    //   // append
    //   sentence.append(" ");
    //   sentence.append(words[i]);
    // }
    // std::ofstream ofs23337("/tmp/cpp_log.txt", std::ios::app);
    // ofs23337 << "sentence2: " << cond_prob << " " << sentence << std::endl;
    // ofs23337.close();

    cond_prob = cond_prob * weight_on_new;
  }

  // return  loge prob
  // NUM_FLT_LOGE = 0.4342944819;
  return cond_prob/NUM_FLT_LOGE;
}

double Scorer::get_sent_log_prob(const std::vector<std::string>& words,
                                 const std::map<std::string, std::string>& funnels,
                                 const std::map<std::string, double> &weights ) {
  std::vector<std::string> sentence;
  if (words.size() == 0) {
    for (size_t i = 0; i < max_order_; ++i) {
      sentence.push_back(START_TOKEN);
    }
  } else {
    for (size_t i = 0; i < max_order_ - 1; ++i) {
      sentence.push_back(START_TOKEN);
    }
    sentence.insert(sentence.end(), words.begin(), words.end());
  }
  sentence.push_back(END_TOKEN);
  return get_log_prob(sentence, funnels, weights);
}

double Scorer::get_log_prob(const std::vector<std::string>& words,
                            const std::map<std::string, std::string>& funnels,
                            const std::map<std::string, double> &weights ) {
  assert(words.size() > max_order_);
  double score = 0.0;
  for (size_t i = 0; i < words.size() - max_order_ + 1; ++i) {
    std::vector<std::string> ngram(words.begin() + i,
                                   words.begin() + i + max_order_);
    // score += get_log_cond_prob(ngram);
    score += get_log_cond_prob(ngram, funnels, weights);
  }
  return score;
}

void Scorer::reset_params(float alpha, float beta) {
  this->alpha = alpha;
  this->beta = beta;
}

std::string Scorer::vec2str(const std::vector<int>& input) {
  std::string word;
  for (auto ind : input) {
    word += char_list_[ind];
  }
  return word;
}

std::vector<std::string> Scorer::split_labels(const std::vector<int>& labels) {
  if (labels.empty()) return {};

  std::string s = vec2str(labels);
  std::vector<std::string> words;
  if (is_character_based_) {
    words = split_utf8_str(s);
  } else {
    words = split_str(s, " ");
  }
  return words;
}

void Scorer::set_char_map(const std::vector<std::string>& char_list) {
  char_list_ = char_list;
  char_map_.clear();

  for (size_t i = 0; i < char_list_.size(); i++) {
    if (char_list_[i] == " ") {
      SPACE_ID_ = i;
    }
    // The initial state of FST is state 0, hence the index of chars in
    // the FST should start from 1 to avoid the conflict with the initial
    // state, otherwise wrong decoding results would be given.
    char_map_[char_list_[i]] = i + 1;
  }
}

std::vector<std::string> Scorer::make_ngram(PathTrie* prefix) {
  std::vector<std::string> ngram;
  PathTrie* current_node = prefix;
  PathTrie* new_node = nullptr;

  for (int order = 0; order < max_order_; order++) {
    std::vector<int> prefix_vec;
    std::vector<int> prefix_steps;

    if (is_character_based_) {
      new_node = current_node->get_path_vec(prefix_vec, prefix_steps, -1, 1);
      current_node = new_node;
    } else {
      new_node = current_node->get_path_vec(prefix_vec, prefix_steps, SPACE_ID_);
      current_node = new_node->parent;  // Skipping spaces
    }

    // reconstruct word
    std::string word = vec2str(prefix_vec);

    ngram.push_back(word);

    if (new_node->character == -1) {
      // No more spaces, but still need order
      for (int i = 0; i < max_order_ - order - 1; i++) {
        ngram.push_back(START_TOKEN);
      }
      break;
    }
  }
  std::reverse(ngram.begin(), ngram.end());
  return ngram;
}

// init dictionary
void Scorer::fill_dictionary(bool add_space) {

  // http://www.openfst.org/twiki/bin/view/FST/FstQuickTour
  fst::StdVectorFst dictionary;

  // // For each unigram convert to ints and put in trie
  int dict_size = 0;
  for (const auto& word : vocabulary_) {

    bool added = add_word_to_dictionary(
        word, char_map_, add_space, SPACE_ID_ + 1, &dictionary);
    dict_size += added ? 1 : 0;
  }

  // this->orig_dictionary = orig_dictionary;

  // // can  i add word to dictinary manually ?
  // // > it worked!!!
  // bool added = add_word_to_dictionary(
  //     "きみしま", char_map_, add_space, SPACE_ID_ + 1, &dictionary);
  // dict_size += added ? 1 : 0;

  dict_size_ = dict_size;


  // ///// tes ok
  // bool added2 = add_word_to_dictionary(
  //     "だこく", char_map_, add_space, SPACE_ID_ + 1, &dictionary);
  // if (added2 == true) {
  //   dict_size_ = dict_size_ + 1;
  // }
  // /////


  this->orig_dictionary = dictionary;

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

  this->dictionary = new_dict;

  // // test of adding vocab
  // add_vocab(true, "だこく");
}


// add vocab to dictionary
int Scorer::add_vocab(bool add_space, std::string vocab) {


  bool added3 = add_word_to_dictionary(
      vocab, char_map_, add_space, SPACE_ID_ + 1, &this->orig_dictionary);
  if (added3 == true) {

    // // show log
    // std::ofstream ofs23334("/tmp/cpp_log.txt", std::ios::app);
    // ofs23334 << "added3 is true: " << vocab << std::endl;
    // ofs23334.close();

    dict_size_ = dict_size_ + 1;

  }

  // this->orig_dictionary = orig_dictionary;

  /* Simplify FST

   * This gets rid of "epsilon" transitions in the FST.
   * These are transitions that don't require a string input to be taken.
   * Getting rid of them is necessary to make the FST determinisitc, but
   * can greatly increase the size of the FST
   */
  fst::RmEpsilon(&this->orig_dictionary);
  fst::StdVectorFst* new_dict = new fst::StdVectorFst;

  /* This makes the FST deterministic, meaning for any string input there's
   * only one possible state the FST could be in.  It is assumed our
   * dictionary is deterministic when using it.
   * (lest we'd have to check for multiple transitions at each state)
   */
  fst::Determinize(this->orig_dictionary, new_dict);

  /* Finds the simplest equivalent fst. This is unnecessary but decreases
   * memory usage of the dictionary
   */
  fst::Minimize(new_dict);

  if (this->dictionary != nullptr) {
    delete static_cast<fst::StdVectorFst*>(dictionary);
  }

  this->dictionary = new_dict;

  return 1;
}


// add vocabs to dictionary
int Scorer::add_vocabs(bool add_space, const std::vector<std::string> &vocabs) {

  // add vocabs into dictionary
  for (auto item : vocabs) {
    bool added3 = add_word_to_dictionary(
        item, char_map_, add_space, SPACE_ID_ + 1, &this->orig_dictionary);
    if (added3 == true) {
      dict_size_ = dict_size_ + 1;
    }
  }

  // for (const auto& [key, value] : funnels){
  //     bool added3 = add_word_to_dictionary(
  //         key, char_map_, add_space, SPACE_ID_ + 1, &this->orig_dictionary);
  //     if (added3 == true) {
  //       dict_size_ = dict_size_ + 1;
  //     }
  // }

  /* Simplify FST

   * This gets rid of "epsilon" transitions in the FST.
   * These are transitions that don't require a string input to be taken.
   * Getting rid of them is necessary to make the FST determinisitc, but
   * can greatly increase the size of the FST
   */
  fst::RmEpsilon(&this->orig_dictionary);
  fst::StdVectorFst* new_dict = new fst::StdVectorFst;

  /* This makes the FST deterministic, meaning for any string input there's
   * only one possible state the FST could be in.  It is assumed our
   * dictionary is deterministic when using it.
   * (lest we'd have to check for multiple transitions at each state)
   */
  fst::Determinize(this->orig_dictionary, new_dict);

  /* Finds the simplest equivalent fst. This is unnecessary but decreases
   * memory usage of the dictionary
   */
  fst::Minimize(new_dict);

  if (this->dictionary != nullptr) {
    delete static_cast<fst::StdVectorFst*>(dictionary);
  }

  this->dictionary = new_dict;

  return 1;
}
