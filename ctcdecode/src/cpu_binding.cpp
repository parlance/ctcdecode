#ifdef TORCH_BINDING
#include <iostream>
#include "ctc_beam_entry.h"
#include "ctc_beam_scorer.h"
#include "ctc_beam_search.h"
#include "ctc_labels.h"
#include "ctc_decoder.h"
#include "util/status.h"
#include "TH.h"
#include "cpu_binding.h"
#include "ctc_beam_scorer_dict.h"

#ifdef INCLUDE_KENLM
#include "ctc_beam_scorer_klm.h"
#include "ctc_trie_node.h"
#include "lm/model.hh"
#endif

namespace pytorch {
  using pytorch::ctc::Labels;
  using pytorch::ctc::Status;
  using pytorch::ctc::DictBeamScorer;
  using pytorch::ctc::ctc_beam_search::DictBeamState;

  #ifdef INCLUDE_KENLM
  using pytorch::ctc::KenLMBeamScorer;
  using pytorch::ctc::ctc_beam_search::KenLMBeamState;
  typedef lm::base::Model Model;

  lm::WordIndex GetWordIndex(const Model* model, const std::string& word) {
    lm::WordIndex vocab;
    vocab = model->BaseVocabulary().Index(word);
    return vocab;
  }

  float ScoreWord(const Model* model, lm::WordIndex vocab) {
    lm::ngram::State in_state;
    lm::ngram::State out;
    lm::FullScoreReturn full_score_return;

    model->BeginSentenceWrite(&in_state);
    full_score_return = model->BaseFullScore(&in_state, vocab, &out);

    return full_score_return.prob;
  }

  int generate_klm_dict_trie(Labels& labels, const char* kenlm_path, const char* vocab_path, const char* trie_path) {
    lm::ngram::Config config;
    config.load_method = util::POPULATE_OR_READ;
    Model* model = lm::ngram::LoadVirtual(kenlm_path, config);
    ctc::TrieNode root(labels.GetSize());

    std::ifstream ifs;
    ifs.open(vocab_path, std::ifstream::in);

    if (!ifs.is_open()) {
      std::cout << "unable to open vocabulary" << std::endl;
      return -1;
    }

    std::ofstream ofs;
    ofs.open(trie_path);

    std::string word;
    while (ifs >> word) {
      lm::WordIndex vocab = GetWordIndex(model, word);
      float unigram_score = ScoreWord(model, vocab);
      std::wstring wide_word;
      utf8::utf8to16(word.begin(), word.end(), std::back_inserter(wide_word));
      root.Insert(wide_word.c_str(), [&labels](wchar_t c) {
                    return labels.GetLabel(c);
                  }, vocab, unigram_score);
    }

    root.WriteToStream(ofs);
    ifs.close();
    ofs.close();
    delete model;
    return 0;
  }
  #endif

  int generate_dict_trie(Labels& labels, const char* vocab_path, const char* trie_path) {
    std::ifstream ifs;
    ifs.open(vocab_path, std::ifstream::in);

    ctc::TrieNode root(labels.GetSize());

    if (!ifs.is_open()) {
      std::cout << "unable to open vocabulary" << std::endl;
      return -1;
    }

    std::ofstream ofs;
    ofs.open(trie_path);

    std::string word;
    int i = 0;
    while (ifs >> word) {
      std::wstring wide_word;
      utf8::utf8to16(word.begin(), word.end(), std::back_inserter(wide_word));
      root.Insert(wide_word.c_str(), [&labels](wchar_t c) {
                    return labels.GetLabel(c);
                  }, i++, 0);
    }
    root.WriteToStream(ofs);
    ifs.close();
    ofs.close();
    return 0;
  }

  extern "C"
  {
    void* get_kenlm_scorer(const wchar_t* label_str, int labels_size, int space_index, int blank_index,
                           const char* lm_path, const char* trie_path) {
      #ifdef INCLUDE_KENLM
      Labels* labels = new Labels(label_str, labels_size, blank_index, space_index);
      ctc::KenLMBeamScorer *beam_scorer = new ctc::KenLMBeamScorer(labels, lm_path, trie_path);
      return static_cast<void*>(beam_scorer);
      #else
      return nullptr;
      #endif
    }

    void free_kenlm_scorer(void* kenlm_scorer) {
      #ifdef INCLUDE_KENLM
      ctc::KenLMBeamScorer* beam_scorer = static_cast<ctc::KenLMBeamScorer*>(kenlm_scorer);
      delete beam_scorer;
      #endif
      return;
    }

    void* get_dict_scorer(const wchar_t* label_str, int labels_size, int space_index, int blank_index,
                          const char* trie_path) {
      Labels* labels = new Labels(label_str, labels_size, blank_index, space_index);
      ctc::DictBeamScorer *beam_scorer = new ctc::DictBeamScorer(labels, trie_path);
      return static_cast<void*>(beam_scorer);
    }

    void set_kenlm_scorer_lm_weight(void *scorer, float weight) {
      #ifdef INCLUDE_KENLM
      ctc::KenLMBeamScorer *beam_scorer = static_cast<ctc::KenLMBeamScorer *>(scorer);
      beam_scorer->SetLMWeight(weight);
      #endif
    }

    void set_kenlm_scorer_wc_weight(void *scorer, float weight) {
      #ifdef INCLUDE_KENLM
      ctc::KenLMBeamScorer *beam_scorer = static_cast<ctc::KenLMBeamScorer *>(scorer);
      beam_scorer->SetWordCountWeight(weight);
      #endif
    }

    void set_kenlm_min_unigram_weight(void *scorer, float weight) {
      #ifdef INCLUDE_KENLM
      ctc::KenLMBeamScorer *beam_scorer = static_cast<ctc::KenLMBeamScorer *>(scorer);
      beam_scorer->SetMinimumUnigramProbability(weight);
      #endif
    }

    void set_dict_min_unigram_weight(void *scorer, float weight) {
      ctc::DictBeamScorer *beam_scorer = static_cast<ctc::DictBeamScorer *>(scorer);
      beam_scorer->SetMinimumUnigramProbability(weight);
    }

    void set_label_selection_parameters(void *decoder, int label_selection_size, float label_selection_margin) {
      ctc::CTCBeamSearchDecoder<> *beam_decoder = static_cast<ctc::CTCBeamSearchDecoder<> *>(decoder);
      beam_decoder->SetLabelSelectionParameters(label_selection_size, label_selection_margin);
    }

    void* get_base_scorer() {
      ctc::CTCBeamSearchDecoder<>::DefaultBeamScorer *beam_scorer = new ctc::CTCBeamSearchDecoder<>::DefaultBeamScorer();
      return static_cast<void *>(beam_scorer);
    }

    void* get_ctc_beam_decoder(int num_classes, int top_paths, int beam_width, int blank_index, void *scorer, DecodeType type) {
      switch (type) {
        case CTC:
        {
          ctc::CTCBeamSearchDecoder<>::DefaultBeamScorer *beam_scorer = static_cast<ctc::CTCBeamSearchDecoder<>::DefaultBeamScorer *>(scorer);
          ctc::CTCBeamSearchDecoder<> *decoder = new ctc::CTCBeamSearchDecoder<>
              (num_classes, beam_width, beam_scorer, blank_index);
          return static_cast<void *>(decoder);
        }
        case CTC_DICT:
        {
          ctc::DictBeamScorer *beam_scorer = static_cast<ctc::DictBeamScorer *>(scorer);
          ctc::CTCBeamSearchDecoder<DictBeamState> *decoder = new ctc::CTCBeamSearchDecoder<DictBeamState>
              (num_classes, beam_width, beam_scorer, blank_index);
          return static_cast<void *>(decoder);
        }
        #ifdef INCLUDE_KENLM
        case CTC_KENLM:
        {
          ctc::KenLMBeamScorer *beam_scorer = static_cast<ctc::KenLMBeamScorer*>(scorer);
          ctc::CTCBeamSearchDecoder<KenLMBeamState> *decoder = new ctc::CTCBeamSearchDecoder<KenLMBeamState>
              (num_classes, beam_width, beam_scorer, blank_index);
          return static_cast<void *>(decoder);
        }
        #endif
      }
      return nullptr;
    }

    int ctc_beam_decode(void *void_decoder, DecodeType type, THFloatTensor *th_probs, THIntTensor *th_seq_len, THIntTensor *th_output,
                        THFloatTensor *th_scores, THIntTensor *th_out_len, THIntTensor *th_alignments, THFloatTensor *th_char_probs)
    {
      const int64_t max_time = THFloatTensor_size(th_probs, 0);
      const int64_t batch_size = THFloatTensor_size(th_probs, 1);
      const int64_t num_classes = THFloatTensor_size(th_probs, 2);
      const int64_t top_paths = THIntTensor_size(th_output, 0);

      // convert tensors to something the beam scorer can use
      // sequence length
      int* seq_len_ptr = THIntTensor_data(th_seq_len);
      ptrdiff_t seq_len_offset = THIntTensor_storageOffset(th_seq_len);
      Eigen::Map<const Eigen::ArrayXi> seq_len(seq_len_ptr + seq_len_offset, batch_size);

      // input logits
      float* probs_ptr = THFloatTensor_data(th_probs);
      ptrdiff_t probs_offset = THFloatTensor_storageOffset(th_probs);
      const int64_t probs_stride_0 = THFloatTensor_stride(th_probs, 0);
      const int64_t probs_stride_1 = THFloatTensor_stride(th_probs, 1);
      const int64_t probs_stride_2 = THFloatTensor_stride(th_probs, 2);
      std::vector<Eigen::Map<Eigen::MatrixXf, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>> inputs;
      for (int t=0; t < max_time; ++t) {
        inputs.emplace_back(probs_ptr + probs_offset + (t*probs_stride_0), batch_size, num_classes, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(probs_stride_2, probs_stride_1));
      }

      // prepare/initialize output variables
      // paths, batches, class
      std::vector<ctc::CTCDecoder::Output> outputs(top_paths);
      for (ctc::CTCDecoder::Output& output : outputs) {
        output.resize(batch_size);
      }
      std::vector<ctc::CTCDecoder::Output> alignments(top_paths);
      for (ctc::CTCDecoder::Output& alignment : alignments) {
        alignment.resize(batch_size);
      }
      std::vector<ctc::CTCDecoder::CharProbability> char_probs(top_paths);
      for (ctc::CTCDecoder::CharProbability& char_ : char_probs) {
        char_.resize(batch_size);
      }
      float score[batch_size][top_paths];
      memset(score, 0.0, batch_size*top_paths*sizeof(int));
      Eigen::Map<Eigen::MatrixXf> *scores;

      // TODO: this is ugly -- can we better leverage generics somehow?
      switch (type) {
        case CTC:
          {
            ctc::CTCBeamSearchDecoder<> *decoder = static_cast<ctc::CTCBeamSearchDecoder<> *>(void_decoder);
            scores = new Eigen::Map<Eigen::MatrixXf>(&score[0][0], batch_size, decoder->GetBeamWidth());
            Status stat = decoder->Decode(seq_len, inputs, &outputs, scores, &alignments, &char_probs);
            if (!stat.ok()) {
              return 0;
            }
          }
          break;
          case CTC_DICT:
          {
            ctc::CTCBeamSearchDecoder<DictBeamState> *decoder = static_cast<ctc::CTCBeamSearchDecoder<DictBeamState> *>(void_decoder);
            scores = new Eigen::Map<Eigen::MatrixXf>(&score[0][0], batch_size, decoder->GetBeamWidth());
            Status stat = decoder->Decode(seq_len, inputs, &outputs, scores, &alignments, &char_probs);
            if (!stat.ok()) {
              return 0;
            }
          }
          break;
        #ifdef INCLUDE_KENLM
        case CTC_KENLM:
          {
            ctc::CTCBeamSearchDecoder<KenLMBeamState> *decoder = static_cast<ctc::CTCBeamSearchDecoder<KenLMBeamState> *>(void_decoder);
            scores = new Eigen::Map<Eigen::MatrixXf>(&score[0][0], batch_size, decoder->GetBeamWidth());
            Status stat = decoder->Decode(seq_len, inputs, &outputs, scores, &alignments, &char_probs);
            if (!stat.ok()) {
              return 0;
            }
          }
          break;
        #endif
      }

      std::vector<float> log_probs;
      for (int p=0; p < top_paths; ++p) {
        int64_t max_decoded = 0;
        int64_t offset = 0;
        for (int b=0; b < batch_size; ++b) {
          auto& p_batch = outputs[p][b];
          auto& alignment_batch = alignments[p][b];
          auto& char_prob_batch = char_probs[p][b];
          int64_t num_decoded = p_batch.size();

          max_decoded = std::max(max_decoded, num_decoded);
          THIntTensor_set2d(th_out_len, p, b, num_decoded);
          for (int64_t t=0; t < num_decoded; ++t) {
            // TODO: this could be more efficient (significant pointer arithmetic every time currently)
            THIntTensor_set3d(th_output, p, b, t, p_batch[t]);
            THIntTensor_set3d(th_alignments, p, b, t, alignment_batch[t]);
            THFloatTensor_set3d(th_char_probs, p, b, t, char_prob_batch[t]);
            THFloatTensor_set2d(th_scores, p, b, (*scores)(b, p));
          }
        }
      }
      delete scores;
      return 1;
    }

    int generate_lm_dict(const wchar_t* label_str, int size, int blank_index, int space_index,
                         const char* lm_path, const char* dictionary_path, const char* output_path) {
        #ifdef INCLUDE_KENLM
        Labels labels(label_str, size, blank_index, space_index);
        return generate_klm_dict_trie(labels, lm_path, dictionary_path, output_path);
        #else
        return -1;
        #endif
    }

    int generate_dict(const wchar_t* label_str, int size, int blank_index, int space_index,
                         const char* dictionary_path, const char* output_path) {
        Labels labels(label_str, size, blank_index, space_index);
        return generate_dict_trie(labels, dictionary_path, output_path);
    }


    int kenlm_enabled() {
      #ifdef INCLUDE_KENLM
      return 1;
      #else
      return 0;
      #endif
    }
  }
}
#endif
