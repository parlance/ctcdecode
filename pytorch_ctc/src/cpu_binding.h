typedef enum {
  CTC,
  CTC_KENLM
} DecodeType ;


/* scorers */
void* get_kenlm_scorer(const wchar_t* label_str, int labels_size, int space_index, int blank_index,
                       const char* lm_path, const char* trie_path);
void* get_base_scorer();


/* decoders */
void* get_ctc_beam_decoder(int num_classes, int batch_size, int top_paths, int beam_width, int blank_index,
                           int merge_repeated, void *scorer, DecodeType type);


/* run decoding */
int ctc_beam_decode(THFloatTensor *probs, THIntTensor *seq_len, THIntTensor *output,
                    THFloatTensor *scores, THIntTensor *th_out_len,
                    void *decoder, DecodeType type);


/* utilities */
int generate_lm_trie(const wchar_t* labels, int size, int blank_index, int space_index,
                     const char* lm_path, const char* dictionary_path, const char* output_path);
