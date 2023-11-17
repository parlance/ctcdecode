black --line-length 100 ctcdecode/ tests/ setup.py

isort ctcdecode/ tests/

clang-format -i ctcdecode/**/*.cpp ctcdecode/src/*.h tools/*