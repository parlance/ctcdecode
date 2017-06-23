int ctc_beam_decode(THFloatTensor *probs, THIntTensor *seq_len, THIntTensor *output,
                    THFloatTensor *scores, THIntTensor *th_out_len, int top_paths,
                    int beam_width, int blank_index, int merge_repeated,
                    const wchar_t* label_str, int labels_size, int space_index, const char* lm_path, const char* trie_path);

int generate_lm_trie(const wchar_t* labels, int size, int blank_index, int space_index,
                     const char* lm_path, const char* dictionary_path, const char* output_path);
