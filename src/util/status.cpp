#include "status.h"
#include <stdio.h>
#include <iostream>
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

    string Status::ToString() const {
      if (state_ == nullptr) {
        return "OK";
      } else {
        char tmp[30];
        const char* type;
        switch (code()) {
          case pytorch::ctc::Code::CANCELLED:
            type = "Cancelled";
            break;
          case pytorch::ctc::Code::INVALID_ARGUMENT:
            type = "Invalid argument";
            break;
          case pytorch::ctc::Code::FAILED_PRECONDITION:
            type = "Failed precondition";
            break;
          case pytorch::ctc::Code::OUT_OF_RANGE:
            type = "Out of range";
            break;
          default:
            snprintf(tmp, sizeof(tmp), "Unknown code(%d)",
                     static_cast<int>(code()));
            type = tmp;
            break;
        }
        string result(type);
        result += ": " + state_->msg;
        return result;
      }
    }

    std::ostream& operator<<(std::ostream& os, const Status& x) {
      os << x.ToString();
      return os;
    }

    namespace errors {
      Status InvalidArgument(string msg) { return Status(Code::INVALID_ARGUMENT, msg); }
      Status FailedPrecondition(string msg) { return Status(Code::FAILED_PRECONDITION, msg); }
    }
  }
}
