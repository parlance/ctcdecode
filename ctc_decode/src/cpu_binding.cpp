
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
    extern int ctc_beam_search(THFloatTensor *probs) {
      return 1;
    }
}
