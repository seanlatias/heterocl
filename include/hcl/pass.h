/*!
 *  Copyright (c) 2021 by Contributors
 * \file ast.h
 */

#ifndef HCL_PASS_H_
#define HCL_PASS_H_

#include "ast/ast.h"
#include "tvm/buffer.h"
#include "tvm/ir.h"
#include "tvm/expr.h"
#include "tvm/operation.h"

#include "mlir/IR/BuiltinOps.h"


namespace hcl {

using namespace ast;

MLIRModule ast_to_hclir(ASTStmt module);

TVM::Stmt hclir_to_tvm(MLIRModule module, TVM::Array<TVM::Buffer> extern_buffer);

}  // namespace hcl

#endif  // HCL_PASS_H_

