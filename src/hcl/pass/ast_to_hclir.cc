/*!
 *  Copyright (c) 2021 by Contributors
 * \file ast_to_hclir.cc
 */

#include "hcl/ast/ast.h"
#include "hcl/dialect/HCLIR/HCLIRDialect.h"
#include "hcl/pass.h"
#include "tvm/api_registry.h"
#include "tvm/expr.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

namespace {

/*
class GenHCLIRDialect {
 public:
  GenHCLIRDialect(mlir::MLIRContext &context) : builder_(&context) {}

  mlir::ModuleOp gen_ir(ASTStmt module) {
    module_ = mlir::ModuleOp::create(builder_.getUnknownLoc());
    return module_;
  }

 private:
  mlir::ModuleOp module_;
  mlir::OpBuilder builder_;
};
*/

}  // namespace

namespace hcl {

void ast_to_hclir(ASTStmt module) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::hclir::HCLIRDialect>();
  //GenHCLIRDialect(context).gen_ir(module);
}

TVM_REGISTER_API("ir_pass.AST2HCL")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      ast_to_hclir(args[0]);
    });

}  // namespace hcl




