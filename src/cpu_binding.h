int ctc_beam_decode(THFloatTensor *probs, THIntTensor *seq_len, THIntTensor *output,
                    THFloatTensor *scores, int num_classes, int beam_width,
                    int batch_size, int merge_repeated);
