### Build FST tool

This is a c++ program to build fst for the given lexicon files.

Run the below steps for creating the build and constructing the FST.

```bash

cd ..
bash build.sh
./build/build_fst --vocab-path <path/to/labels.txt> --lexicon-paths <path to multiple lexicon files separated by space> --output-path <path to output file> --freq-threshold 30[Optional] --fst-path <path to a fst file>[Optional]

```

- Vocab path - Each line in the file should contain single label. Example file is provided in   `tests/cpp/fixtures` directory
- Lexicon paths - Path to single/multiple lexicon files. Each line in the file should be like this: <frequency> <word> <tokens separated by space>
- Frequency threshold - Words having frequency greater than or equal to this threshold will be considered while constructing the FST. (Default is -1 i.e all are considered)
- Fst path - If a fst file is provided, then the given lexicon words will be added on top of this FST file. 
- Output path - Path to output file. Two output files will be generated. One with `.opt` extension contains optimized FST and other contains unoptimized. 

For more information, run `./build/build_fst --help`