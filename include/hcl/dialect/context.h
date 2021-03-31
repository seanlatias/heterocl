/*!
 *  Copyright (c) 2021 by Contributors
 * \file context.h
 */
#ifndef HCL_DIALECT_CONTEXT_H_
#define HCL_DIALECT_CONTEXT_H_

#include "mlir/IR/MLIRContext.h"

namespace hcl {

class IRContext {
 public:
  static mlir::MLIRContext* context;
};

}  // namespace hcl

#endif  // HCL_DIALECT_CONTEXT_H_
