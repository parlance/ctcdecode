#include "status.h"
#include <stdio.h>
#include <string>
#include <cassert>

using namespace std;

namespace pytorch {
  namespace ctc {
    Status::Status(pytorch::ctc::Code code, string msg) {
      assert(code != pytorch::ctc::OK);
      state_ = new State;
      state_->code = code;
      state_->msg = msg;
    }

    const string& Status::empty_string() {
      static string* empty = new string;
      return *empty;
    }
    namespace errors {
      Status InvalidArgument(string msg) { return Status(Code::INVALID_ARGUMENT, msg); }
    }
  }
}
