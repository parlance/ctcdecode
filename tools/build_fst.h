#ifndef BUILD_FST_H
#define BUILD_FST_H

#include "fst/fst.h"
#include "fst/fstlib.h"

std::vector<std::string> get_bpe_vocab(const std::string vocab_path);

std::unordered_map<std::string, int> get_char_map(const std::vector<std::string>& labels);

fst::StdVectorFst* read_fst(const std::string output_path);

void write_fst(fst::StdVectorFst* dictionary, const std::string output_path);

void optimize_fst(fst::StdVectorFst* dictionary);

bool add_word_to_fst(const std::vector<std::string>& characters,
                     const std::unordered_map<std::string, int>& char_map,
                     fst::StdVectorFst* dictionary,
                     fst::StdVectorFst::StateId current_state);

int parse_lexicon_and_add_to_fst(const std::string& lexicon_path,
                                 fst::StdVectorFst* dictionary,
                                 const std::unordered_map<std::string, int>& char_map,
                                 const int freq_threshold);

void construct_fst(const std::string vocab_path,
                   const std::vector<std::string>& lexicon_paths,
                   const std::string fst_path,
                   std::string output_path,
                   const int freq_threshold,
                   bool optimize);

#endif
