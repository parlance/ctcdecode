int ctc_beam_decode(THFloatTensor *probs, THIntTensor *seq_len, THIntTensor *output,
                    THFloatTensor *scores, int beam_width, int merge_repeated);

int test_ctc_beam_decode();
