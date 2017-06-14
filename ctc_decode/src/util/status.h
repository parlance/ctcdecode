#include <string>

#ifndef PYTORCH_CONTRIB_CTC_CTC_STATUS_H_
#define PYTORCH_CONTRIB_CTC_CTC_STATUS_H_

using namespace std;

namespace pytorch {
  namespace ctc {
    enum Code {
      OK,
      CANCELLED,
      INVALID_ARGUMENT,
      FAILED_PRECONDITION,
      OUT_OF_RANGE
    };

    class Status {
    public:
      Status() : state_(NULL) {}
      ~Status() { delete state_; }

      Status(pytorch::ctc::Code code, string msg);

      static Status OK() { return Status(); }

      bool ok() const { return (state_ == nullptr); }

      pytorch::ctc::Code code() const {
        return ok() ? pytorch::ctc::OK : state_->code;
      }

      const string& error_message() const {
        return ok() ? empty_string() : state_->msg;
      }

    private:
      static const string& empty_string();
      struct State {
        pytorch::ctc::Code code;
        string msg;
      };
      // OK status has a `NULL` state_.  Otherwise, `state_` points to
      // a `State` structure containing the error code and message(s)
      State* state_;
    };

    namespace errors {
      Status InvalidArgument(string msg);
    }
  }
}

#endif // PYTORCH_CONTRIB_CTC_CTC_STATUS_H_
