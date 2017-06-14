
#include <iostream>
#include "ctc_beam_entry.h"
#include "ctc_beam_scorer.h"
#include "ctc_beam_search.h"
#include "ctc_decoder.h"
#include "TH.h"

class User
{
    std::string name;
    public:
        User(char *name):name(name) {}
        User(std::string &name):name(name) {}

        std::string greet() { return "hello, " + name; }
};

void hello(char *name)
{
    User user(name);
    std::cout << user.greet() << std::endl;
}

extern "C"
{
    int ctc_beam_decode(THFloatTensor *probs, THIntTensor *seq_len, THIntTensor *output,
                        THFloatTensor *scores, int num_classes, int beam_width,
                        int batch_size, int merge_repeated)
    {
      return 1;
    }
}
