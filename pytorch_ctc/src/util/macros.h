#ifndef PYTORCH_CONTRIB_CTC_CTC_MACROS_H_
#define PYTORCH_CONTRIB_CTC_CTC_MACROS_H_

// A macro to disallow the copy constructor and operator= functions
// This is usually placed in the private: declarations for a class.
#define TF_DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName&) = delete;         \
  void operator=(const TypeName&) = delete

#endif
