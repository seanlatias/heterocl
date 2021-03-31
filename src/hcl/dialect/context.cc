/*!
 *  Copyright (c) 2021 by Contributors
 * \file hclir_to_tvm.cc
 */
#include "hcl/dialect/context.h"

namespace hcl {

mlir::MLIRContext* IRContext::context = new mlir::MLIRContext();

}  // namespace hcl
