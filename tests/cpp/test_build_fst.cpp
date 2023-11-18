#include <gtest/gtest.h>
#include <iostream>

#include "build_fst.h"

// create a sample fst from the lexicon file and
// compare it with the expected fst
TEST(BuildFstTest, TestBuildFst)
{
    std::vector<std::string> lexicon_paths = { std::string(TEST_FIXTURES_DIR) + "/lexicon.txt" };
    std::string label_path = std::string(TEST_FIXTURES_DIR) + "/vocab.txt";
    std::string expected_fst_path = std::string(TEST_FIXTURES_DIR) + "/expected_fst.fst";
    std::string output_fst_path = ::testing::TempDir() + "/test_output.fst";
    int freq_threshold = 0;

    construct_fst(label_path, lexicon_paths, "", output_fst_path, freq_threshold, false);

    auto output_fst = read_fst(output_fst_path);
    auto expected_fst = read_fst(expected_fst_path);
    EXPECT_EQ(output_fst->NumStates(), expected_fst->NumStates());
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}