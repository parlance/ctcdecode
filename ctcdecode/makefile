OBJ_DIR := ./obj
SRC_FILES := $(wildcard ./src/*.cpp)
SRC_FILES2 := $(wildcard ../third_party/kenlm/util/*.cc)
SRC_FILES2 += $(wildcard ../third_party/kenlm/lm/*.cc)
SRC_FILES2 += $(wildcard ../third_party/kenlm/util/double-conversion/*.cc)
SRC_FILES2 += $(wildcard ../third_party/openfst-1.6.7/src/lib/*.cc)
TESTS := $(wildcard ../third_party/kenlm/util/*test.cc)
TESTS += $(wildcard ../third_party/kenlm/util/*main.cc)
TESTS += $(wildcard ../third_party/kenlm/lm/*test.cc)
TESTS += $(wildcard ../third_party/kenlm/lm/*main.cc)

SRC_FILES2 := $(filter-out $(TESTS),$(SRC_FILES2))

OBJ_FILES := $(SRC_FILES:.cpp=.o)
OBJ_FILES += $(SRC_FILES2:.cc=.o)


LDFLAGS := -I../third_party/boost_1_67_0 -fPIC -O3 -std=c++14 -shared -ldl
CPPFLAGS := -I../third_party/openfst-1.6.7/src/include -I../third_party/kenlm -I../third_party/ThreadPool -I../third_party/utf8 -I../third_party/boost_1_67_0 -DKENLM_MAX_ORDER=6 -DINCLUDE_KENLM -fPIC -O3 -std=c++14 -shared -ldl

NativeCTCBeamDecoder.so: $(OBJ_FILES)
#	echo $(OBJ_FILES)
	echo $(TESTS)
	 g++ $(LDFLAGS) -o $@ $^

clean:
	rm $(OBJ_FILES)
.o: 
	 g++ $(CPPFLAGS)  -c -o $@ $<