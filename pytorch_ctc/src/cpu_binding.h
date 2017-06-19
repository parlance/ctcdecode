int ctc_beam_decode(THFloatTensor *probs, THIntTensor *seq_len, THIntTensor *output,
                    THFloatTensor *scores, THIntTensor *th_out_len, int top_paths,
                    int beam_width, int blank_index, int merge_repeated);
