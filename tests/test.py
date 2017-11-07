import unittest

import torch
import numpy as np
import ctcdecode
from torch.nn.functional import log_softmax
from torch.autograd import Variable


class CTCDecodeTests(unittest.TestCase):
    def test_simple_decode(self):
        aa = torch.FloatTensor(
            np.array([[[1.0, 0.0]], [[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 0.0]], [[1.0, 0.0]]], dtype=np.float32)).log()
        seq_len = torch.IntTensor(np.array([5], dtype=np.int32))

        labels = "A_"
        scorer = ctcdecode.Scorer()
        decoder_nomerge = ctcdecode.CTCBeamDecoder(scorer, labels, blank_index=1, space_index=-1, top_paths=1,
                                                     beam_width=1)

        result_nomerge, _, result_nomerge_len, nomerge_alignments, _ = decoder_nomerge.decode(aa, seq_len)
        self.assertEqual(result_nomerge_len[0][0], 2)
        self.assertEqual(result_nomerge.numpy()[0, 0, :result_nomerge_len[0][0]].tolist(), [0, 0])

    def test_simple_decode_different_blank_idx(self):
        aa = torch.FloatTensor(
            np.array([[[0.0, 1.0]], [[0.0, 1.0]], [[1.0, 0.0]], [[0.0, 1.0]], [[0.0, 1.0]]], dtype=np.float32)).log()
        seq_len = torch.IntTensor(np.array([5], dtype=np.int32))

        labels = "_A"
        scorer = ctcdecode.Scorer()
        decoder_nomerge = ctcdecode.CTCBeamDecoder(scorer, labels, blank_index=0, space_index=-1, top_paths=1,
                                                     beam_width=1)

        result_nomerge, _, result_nomerge_len, nomerge_alignments, _ = decoder_nomerge.decode(aa, seq_len)
        self.assertEqual(result_nomerge_len[0][0], 2)
        self.assertEqual(result_nomerge.numpy()[0, 0, :result_nomerge_len[0][0]].tolist(), [1, 1])

    def test_ctc_decoder_beam_search(self):
        depth = 6
        seq_len_0 = 5
        input_prob_matrix_0 = np.asarray(
            [
                [0.30999, 0.309938, 0.0679938, 0.0673362, 0.0708352, 0.173908],
                [0.215136, 0.439699, 0.0370931, 0.0393967, 0.0381581, 0.230517],
                [0.199959, 0.489485, 0.0233221, 0.0251417, 0.0233289, 0.238763],
                [0.279611, 0.452966, 0.0204795, 0.0209126, 0.0194803, 0.20655],
                [0.51286, 0.288951, 0.0243026, 0.0220788, 0.0219297, 0.129878],
                # Random entry added in at time=5
                [0.155251, 0.164444, 0.173517, 0.176138, 0.169979, 0.160671]
            ],
            dtype=np.float32)
        # Take the log
        input_log_prob_matrix_0 = np.log(input_prob_matrix_0)

        # len max_time_steps array of batch_size x depth matrices
        inputs = np.array([
                              input_log_prob_matrix_0[t, :][np.newaxis, :] for t in range(seq_len_0)
                              ]  # Pad to max_time_steps = 8
                          + 2 * [np.zeros(
            (1, depth), dtype=np.float32)], dtype=np.float32)

        # batch_size length vector of sequence_lengths
        seq_lens = np.array([seq_len_0], dtype=np.int32)

        th_input = torch.from_numpy(inputs)
        th_seq_len = torch.IntTensor(seq_lens)

        labels = "ABCDE_"
        scorer = ctcdecode.Scorer()
        decoder = ctcdecode.CTCBeamDecoder(scorer, labels, blank_index=5, space_index=-1, top_paths=2, beam_width=2)

        decode_result, scores, decode_len, alignments, _ = decoder.decode(th_input, th_seq_len)

        self.assertEqual(decode_len[0][0], 2)
        self.assertEqual(decode_len[1][0], 3)
        self.assertEqual(decode_result.numpy()[0, 0, :decode_len[0][0]].tolist(), [1, 0])
        self.assertEqual(decode_result.numpy()[1, 0, :decode_len[1][0]].tolist(), [0, 1, 0])
        self.assertEqual(alignments.numpy()[0, 0, :decode_len[0][0]].tolist(), [0, 4])
        self.assertEqual(alignments.numpy()[1, 0, :decode_len[1][0]].tolist(), [0, 2, 4])
        np.testing.assert_almost_equal(scores.numpy(), np.array([[-3.58212], [-3.77783]]), 5)

    def test_ctc_output_probability(self):
        seq_len_0 = 2
        classes = 3
        input_prob_matrix_0 = np.asarray(
            [
                [0.4, 0.00000001, 0.6],
                [0.4, 0.00000001, 0.6]
            ],
            dtype=np.float32
        )
        input_log_prob_matrix_0 = np.log(input_prob_matrix_0)
        inputs = np.array([input_log_prob_matrix_0[t, :][np.newaxis, :] for t in range(seq_len_0)])
        seq_lens = np.array([seq_len_0], dtype=np.int32)

        th_input = torch.from_numpy(inputs)
        th_seq_len = torch.IntTensor(seq_lens)

        labels = "AB_"
        scorer = ctcdecode.Scorer()
        decoder = ctcdecode.CTCBeamDecoder(scorer, labels, blank_index=2, space_index=-1, top_paths=1, beam_width=3)

        decode_result, scores, decode_len, alignments, char_probs = decoder.decode(th_input, th_seq_len)
        self.assertEqual(decode_len[0][0], 1)
        self.assertEqual(decode_result.numpy()[0, 0, :decode_len[0][0]].tolist(), [0])
        self.assertEqual(alignments.numpy()[0, 0, :decode_len[0][0]].tolist(), [1])
        np.testing.assert_almost_equal(scores.numpy(), np.log(np.array([[0.64]])), 5)

    def test_ctc_decoder_beam_search_different_blank_idx(self):
        depth = 6
        seq_len_0 = 5
        input_prob_matrix_0 = np.asarray(
            [
                [0.173908, 0.30999, 0.309938, 0.0679938, 0.0673362, 0.0708352],
                [0.230517, 0.215136, 0.439699, 0.0370931, 0.0393967, 0.0381581],
                [0.238763, 0.199959, 0.489485, 0.0233221, 0.0251417, 0.0233289],
                [0.20655, 0.279611, 0.452966, 0.0204795, 0.0209126, 0.0194803],
                [0.129878, 0.51286, 0.288951, 0.0243026, 0.0220788, 0.0219297],
                # Random entry added in at time=5
                [0.160671, 0.155251, 0.164444, 0.173517, 0.176138, 0.169979]
            ],
            dtype=np.float32)
        # Take the log
        input_log_prob_matrix_0 = np.log(input_prob_matrix_0)

        # len max_time_steps array of batch_size x depth matrices
        inputs = np.array([
                              input_log_prob_matrix_0[t, :][np.newaxis, :] for t in range(seq_len_0)
                              ]  # Pad to max_time_steps = 8
                          + 2 * [np.zeros(
            (1, depth), dtype=np.float32)], dtype=np.float32)

        # batch_size length vector of sequence_lengths
        seq_lens = np.array([seq_len_0], dtype=np.int32)

        th_input = torch.from_numpy(inputs)
        th_seq_len = torch.IntTensor(seq_lens)

        labels = "_ABCDE"
        scorer = ctcdecode.Scorer()
        decoder = ctcdecode.CTCBeamDecoder(scorer, labels, blank_index=0, space_index=-1, top_paths=2, beam_width=2)

        decode_result, scores, decode_len, alignments, char_probs = decoder.decode(th_input, th_seq_len)
        self.assertEqual(decode_len[0][0], 2)
        self.assertEqual(decode_len[1][0], 3)
        self.assertEqual(decode_result.numpy()[0, 0, :decode_len[0][0]].tolist(), [2, 1])
        self.assertEqual(decode_result.numpy()[1, 0, :decode_len[1][0]].tolist(), [1, 2, 1])
        self.assertEqual(alignments.numpy()[0, 0, :decode_len[0][0]].tolist(), [0, 4])
        self.assertEqual(alignments.numpy()[1, 0, :decode_len[1][0]].tolist(), [0, 2, 4])
        np.testing.assert_almost_equal(scores.numpy(), np.array([[-3.58212], [-3.77783]]), 5)

    def test_real_ctc_decode(self):
        data = np.genfromtxt("data/rnnOutput.csv", delimiter=';')[:, :-1]
        inputs = np.array([
                              data[t, :][np.newaxis, :] for t in range(data.shape[0])
                              ])
        # Pad to max_time_steps = 8
        #          + 2 * [-5*np.ones(
        #              (1, 80), dtype=np.float32)], dtype=np.float32)
        seq_lens = np.array([inputs.shape[0]], dtype=np.int32)
        th_input = torch.from_numpy(inputs).type(torch.FloatTensor)
        th_input = log_softmax(Variable(th_input), dim=2).data
        th_seq_len = torch.IntTensor(seq_lens)

        labels = ' !"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_'

        # greedy using beam
        scorer = ctcdecode.Scorer()
        decoder = ctcdecode.CTCBeamDecoder(scorer, labels, blank_index=labels.index('_'),
                                             space_index=labels.index(' '), top_paths=1, beam_width=1)
        decode_result, scores, decode_len, alignments, char_probs = decoder.decode(th_input, th_seq_len)
        txt_result = ''.join([labels[x] for x in decode_result[0][0][0:decode_len[0][0]]])
        self.assertEqual("the fak friend of the fomly hae tC", txt_result)

        # default beam decoding
        decoder = ctcdecode.CTCBeamDecoder(scorer, labels, blank_index=labels.index('_'),
                                             space_index=labels.index(' '), top_paths=1, beam_width=25)
        decode_result, scores, decode_len, alignments, char_probs = decoder.decode(th_input, th_seq_len)
        txt_result = ''.join([labels[x] for x in decode_result[0][0][0:decode_len[0][0]]])
        self.assertEqual("the fak friend of the fomcly hae tC", txt_result)

        # dictionary-based decoding where non-words and words are equiprobable. Equivalent to standard beam decoding
        scorer = ctcdecode.DictScorer(labels, "data/ocr.trie", blank_index=labels.index('_'),
                                        space_index=labels.index(' '))
        scorer.set_min_unigram_weight(0.0)
        decoder = ctcdecode.CTCBeamDecoder(scorer, labels, blank_index=labels.index('_'),
                                             space_index=labels.index(' '), top_paths=1, beam_width=25)
        decode_result, scores, decode_len, alignments, char_probs = decoder.decode(th_input, th_seq_len)
        txt_result = ''.join([labels[x] for x in decode_result[0][0][0:decode_len[0][0]]])
        self.assertEqual("the fak friend of the fomcly hae tC", txt_result)

        # dictionary-based decoding - only dictionary words can be emitted
        scorer = ctcdecode.DictScorer(labels, "data/ocr.trie", blank_index=labels.index('_'),
                                        space_index=labels.index(' '))
        decoder = ctcdecode.CTCBeamDecoder(scorer, labels, blank_index=labels.index('_'),
                                             space_index=labels.index(' '), top_paths=1, beam_width=25)
        decode_result, scores, decode_len, alignments, char_probs = decoder.decode(th_input, th_seq_len)
        txt_result = ''.join([labels[x] for x in decode_result[0][0][0:decode_len[0][0]]])
        self.assertEqual("the fake friend of the family, fake the", txt_result)

        # lm-based decoding
        scorer = ctcdecode.KenLMScorer(labels, "data/bigram.arpa", "data/ocr.trie", blank_index=labels.index('_'),
                                         space_index=labels.index(' '))
        scorer.set_lm_weight(2.0)
        scorer.set_word_weight(0)
        decoder = ctcdecode.CTCBeamDecoder(scorer, labels, blank_index=labels.index('_'),
                                             space_index=labels.index(' '), top_paths=1, beam_width=25)
        decode_result, scores, decode_len, alignments, char_probs = decoder.decode(th_input, th_seq_len)
        txt_result = ''.join([labels[x] for x in decode_result[0][0][0:decode_len[0][0]]])
        self.assertEqual("the fake friend of the family, like the", txt_result)


if __name__ == '__main__':
    unittest.main()
