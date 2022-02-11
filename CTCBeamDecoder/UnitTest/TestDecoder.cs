using System;
using System.Linq;
using System.Text;
using CTCBeamDecoder;
using CTCBeamDecoder.Models;
using Xunit;

namespace UnitTest
{
    public class TestDecoder
    {
        public static readonly string[] Vocabulary = new string[]
        {
            "'", " ", "a", "b", "c", "d", "_"
        };
        public static uint BlankId => (uint)Array.IndexOf(Vocabulary, "_");
        public static readonly uint BeamSize = 20;
        public readonly float[,,] Probs1 = new float[,,]
        {
            {
                { 0.06390443f, 0.21124858f, 0.27323887f, 0.06870235f, 0.0361254f, 0.18184413f, 0.16493624f },
                { 0.03309247f, 0.22866108f, 0.24390638f, 0.09699597f, 0.31895462f, 0.0094893f, 0.06890021f },
                { 0.218104f, 0.19992557f, 0.18245131f, 0.08503348f, 0.14903535f, 0.08424043f, 0.08120984f },
                { 0.12094152f, 0.19162472f, 0.01473646f, 0.28045061f, 0.24246305f, 0.05206269f, 0.09772094f },
                { 0.1333387f, 0.00550838f, 0.00301669f, 0.21745861f, 0.20803985f, 0.41317442f, 0.01946335f },
                { 0.16468227f, 0.1980699f, 0.1906545f, 0.18963251f, 0.19860937f, 0.04377724f, 0.01457421f }
            }
        };
        public readonly float[,,] Probs2 = new float[,,]
        {
            {
                { 0.08034842f, 0.22671944f, 0.05799633f, 0.36814645f, 0.11307441f, 0.04468023f, 0.10903471f },
                { 0.09742457f, 0.12959763f, 0.09435383f, 0.21889204f, 0.15113123f, 0.10219457f, 0.20640612f },
                { 0.45033529f, 0.09091417f, 0.15333208f, 0.07939558f, 0.08649316f, 0.12298585f, 0.01654384f },
                { 0.02512238f, 0.22079203f, 0.19664364f, 0.11906379f, 0.07816055f, 0.22538587f, 0.13483174f },
                { 0.17928453f, 0.06065261f, 0.41153005f, 0.1172041f, 0.11880313f, 0.07113197f, 0.04139363f },
                { 0.15882358f, 0.1235788f, 0.23376776f, 0.20510435f, 0.00279306f, 0.05294827f, 0.22298418f }
            }
        };
        public readonly float[,,] Probs3 = new float[,,]
        {
            {
                { 0.06390443f, 0.21124858f, 0.27323887f, 0.06870235f, 0.0361254f, 0.18184413f, 0.16493624f },
                { 0.03309247f, 0.22866108f, 0.24390638f, 0.09699597f, 0.31895462f, 0.0094893f, 0.06890021f },
                { 0.218104f, 0.19992557f, 0.18245131f, 0.08503348f, 0.14903535f, 0.08424043f, 0.08120984f },
                { 0.12094152f, 0.19162472f, 0.01473646f, 0.28045061f, 0.24246305f, 0.05206269f, 0.09772094f },
                { 0.1333387f, 0.00550838f, 0.00301669f, 0.21745861f, 0.20803985f, 0.41317442f, 0.01946335f },
                { 0.16468227f, 0.1980699f, 0.1906545f, 0.18963251f, 0.19860937f, 0.04377724f, 0.01457421f }
            },
            {
                { 0.08034842f, 0.22671944f, 0.05799633f, 0.36814645f, 0.11307441f, 0.04468023f, 0.10903471f },
                { 0.09742457f, 0.12959763f, 0.09435383f, 0.21889204f, 0.15113123f, 0.10219457f, 0.20640612f },
                { 0.45033529f, 0.09091417f, 0.15333208f, 0.07939558f, 0.08649316f, 0.12298585f, 0.01654384f },
                { 0.02512238f, 0.22079203f, 0.19664364f, 0.11906379f, 0.07816055f, 0.22538587f, 0.13483174f },
                { 0.17928453f, 0.06065261f, 0.41153005f, 0.1172041f, 0.11880313f, 0.07113197f, 0.04139363f },
                { 0.15882358f, 0.1235788f, 0.23376776f, 0.20510435f, 0.00279306f, 0.05294827f, 0.22298418f }
            }
        };

        public static readonly string[] GreedyResult = new string[] { "ac'bdc", "b'da" };
        public static readonly string[] BeamSearchResult = new string[] { "acdc", "b'a", "a a" };
        public static string ModelPath = "test.arpa";

        public static string ConvertToString(DecoderResult result, int index)
        {
            var builder = new StringBuilder();

            for (int i = 0; i < result.OutLens[index, 0]; i++)
            {
                builder.Append(Vocabulary[result.BeamResults[index, 0, i]]);
            }

            return builder.ToString();
        }

        public static float[,,] MergeProbs(params float[][,,] probs)
        {
            var result = new float[probs.Select(x => x.GetLength(0)).Sum(), probs.Select(x => x.GetLength(1)).Max(), probs.Select(x => x.GetLength(2)).Max()];

            int si = 0;

            foreach (var prob in probs)
            {
                for (int i = 0; i < prob.GetLength(0); i++, si++)
                {
                    for (int j = 0; j < prob.GetLength(1); j++)
                    {
                        for (int k = 0; k < prob.GetLength(2); k++)
                        {
                            result[si, j, k] = prob[i, j, k];
                        }
                    }
                }
            }

            return result;
        }

        public static float[,,] LogProbs(float[,,] probs)
        {
            var result = new float[probs.GetLength(0), probs.GetLength(1), probs.GetLength(2)];

            for (int i = 0; i < probs.GetLength(0); i++)
            {
                for (int j = 0; j < probs.GetLength(1); j++)
                {
                    for (int k = 0; k < probs.GetLength(2); k++)
                    {
                        result[i, j, k] = (float)Math.Log(probs[i, j, k]);
                    }
                }
            }

            return result;
        }

        public static float[,,] SliceProbs(float[,,] probs, int secondDimensionalStart, int secondDimensionalEnd)
        {
            var result = new float[probs.GetLength(0), secondDimensionalEnd - secondDimensionalStart, probs.GetLength(2)];

            for (int i = 0; i < probs.GetLength(0); i++)
            {
                for (int j = secondDimensionalStart; j < secondDimensionalEnd; j++)
                {
                    for (int k = 0; k < probs.GetLength(2); k++)
                    {
                        result[i, j - secondDimensionalStart, k] = probs[i, j, k];
                    }
                }
            }

            return result;
        }

        [Fact]
        public void TestBeamSearchDecoder1()
        {
            using var scorer = new DecoderScorer(Vocabulary);
            var decoder = new CTCDecoder(beamWidth: BeamSize, blankId: BlankId);
            var result = decoder.Decode(Probs1, scorer);
            Assert.Equal(BeamSearchResult[0], ConvertToString(result, 0));
        }

        [Fact]
        public void TestBeamSearchDecoder2()
        {
            using var scorer = new DecoderScorer(Vocabulary);
            var decoder = new CTCDecoder(beamWidth: BeamSize, blankId: BlankId);
            var result = decoder.Decode(Probs2, scorer);
            Assert.Equal(BeamSearchResult[1], ConvertToString(result, 0));
        }

        [Fact]
        public void TestBeamSearchDecoder3()
        {
            using var scorer = new DecoderScorer(Vocabulary, "test.arpa");
            var decoder = new CTCDecoder(beamWidth: BeamSize, blankId: BlankId);
            var result = decoder.Decode(Probs2, scorer);
            Assert.Equal(BeamSearchResult[2], ConvertToString(result, 0));
        }

        [Fact]
        public void TestBeamSearchDecoderBatch()
        {
            using var scorer = new DecoderScorer(Vocabulary);
            var decoder = new CTCDecoder(beamWidth: BeamSize, blankId: BlankId, numProcesses: 24);
            var result = decoder.Decode(Probs3, scorer);
            Assert.Equal(BeamSearchResult[0], ConvertToString(result, 0));
            Assert.Equal(BeamSearchResult[1], ConvertToString(result, 1));
        }

        [Fact]
        public void TestBeamSearchDecoderBatchLog()
        {
            using var scorer = new DecoderScorer(Vocabulary);
            var decoder = new CTCDecoder(beamWidth: BeamSize, blankId: BlankId, numProcesses: 24, logProbsInput: true);
            var result = decoder.Decode(LogProbs(Probs3), scorer);
            Assert.Equal(BeamSearchResult[0], ConvertToString(result, 0));
            Assert.Equal(BeamSearchResult[1], ConvertToString(result, 1));
        }

        [Fact]
        public void TestBeamSearchOnlineDecoder()
        {
            using var scorer = new DecoderScorer(Vocabulary, ModelPath);
            var decoder = new CTCDecoder(beamWidth: BeamSize, blankId: BlankId, numProcesses: 24, logProbsInput: true);
            using var state1 = decoder.GetState(scorer);
            using var state2 = decoder.GetState(scorer);

            var ios = new bool[] { true, true };
            var result = decoder.DecodeOnline(LogProbs(MergeProbs(Probs2, Probs2)), new[] { state1, state2 }, ios);

            Assert.Equal(BeamSearchResult[2], ConvertToString(result, 1));
            Assert.Equal(BeamSearchResult[2], ConvertToString(result, 0));
        }

        [Fact]
        public void TestBeamSearchOnlineDecoderNoLm()
        {
            using var scorer = new DecoderScorer(Vocabulary);
            var decoder = new CTCDecoder(beamWidth: BeamSize, blankId: BlankId, numProcesses: 24, logProbsInput: true);
            using var state1 = decoder.GetState(scorer);
            using var state2 = decoder.GetState(scorer);
            var ios = new bool[] { true, true };
            var result = decoder.DecodeOnline(LogProbs(Probs3), new[] { state1, state2 }, ios);
            Assert.Equal(BeamSearchResult[0], ConvertToString(result, 0));
            Assert.Equal(BeamSearchResult[1], ConvertToString(result, 1));
        }

        [Fact]
        public void TestBeamSearchOnlineDecoderWithTwoCalls()
        {
            using var scorer = new DecoderScorer(Vocabulary, ModelPath);
            var decoder = new CTCDecoder(beamWidth: BeamSize, blankId: BlankId, numProcesses: 24, logProbsInput: true);
            using var state1 = decoder.GetState(scorer);
            var result = decoder.DecodeOnline(SliceProbs(LogProbs(Probs2), 0, 2), new[] { state1 }, new[] { false });

            result = decoder.DecodeOnline(SliceProbs(LogProbs(Probs2), 2, Probs2.GetLength(1)), new[] { state1 }, new[] { true });

            Assert.Equal(BeamSearchResult[2], ConvertToString(result, 0));
        }

        [Fact]
        public void TestBeamSearchOnlineDecoderWithTwoCallsNoLM()
        {
            using var scorer = new DecoderScorer(Vocabulary);
            var decoder = new CTCDecoder(beamWidth: BeamSize, blankId: BlankId, numProcesses: 24, logProbsInput: true);
            using var state1 = decoder.GetState(scorer);
            using var state2 = decoder.GetState(scorer);

            decoder.DecodeOnline(SliceProbs(LogProbs(Probs3), 0, 2), new[] { state1, state2 }, new[] { false, false });
            var result = decoder.DecodeOnline(SliceProbs(LogProbs(Probs3), 2, Probs3.GetLength(1)), new[] { state1, state2 }, new[] { true, true });

            Assert.Equal(BeamSearchResult[0], ConvertToString(result, 0));
            Assert.Equal(BeamSearchResult[1], ConvertToString(result, 1));
        }
    }
}
