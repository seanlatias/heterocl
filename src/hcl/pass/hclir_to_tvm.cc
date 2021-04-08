/*!
 *  Copyright (c) 2021 by Contributors
 * \file hclir_to_tvm.cc
 */

#include "tvm/buffer.h"
#include "tvm/ir.h"
#include "tvm/expr.h"
#include "tvm/operation.h"
#include "tvm/api_registry.h"
#include "hcl/ast/ast.h"
#include "hcl/dialect/HCLIR/HCLIROps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"

using namespace TVM;
using namespace TVM::ir;
using namespace mlir::hclir;

namespace {

class GenTVMIR {
 public:
  GenTVMIR(mlir::ModuleOp module) : module_(module) {}

  Stmt generate(Array<Buffer> extern_buffer) {
    auto& block = module_.getRegion().front();
    for (mlir::Operation &op : block.getOperations()) {
      if (auto func_op = mlir::dyn_cast<mlir::FuncOp>(&op)) {
        llvm::errs() << func_op.sym_name();
      }
    }
    return Stmt();
  }

 private:
  mlir::ModuleOp module_;
  std::map <mlir::Value, Buffer> buffers_;

  Type gen_type(mlir::Type type) {
    if (type.isSignedInteger()) {
      return Int(type.getIntOrFloatBitWidth());
    } else if (type.isUnsignedInteger()) {
      return UInt(type.getIntOrFloatBitWidth());
    } else if (type.isF32()) {
      return Float(32);
    }
  }

  Expr gen_expr(mlir::Value val) {
    const mlir::Operation* op = val.getDefiningOp();
    if (mlir::dyn_cast<AddOp>(op)) {
      return gen_expr(val.getDefiningOp<AddOp>());
    }
  }

  Expr gen_expr(AddOp op) {
    Type t = gen_type(op.getType());
    Expr a = gen_expr(op.getOperand(0));
    Expr b = gen_expr(op.getOperand(1));
    return Add::make(a, b);
  }


};

}  // namespace


namespace hcl {

Stmt hclir_to_tvm(ast::MLIRModule mod, Array<Buffer> extern_buffer) {
  mlir::ModuleOp module = mod->module.get();
  return GenTVMIR(module).generate(extern_buffer);
}

TVM_REGISTER_API("ir_pass.HCL2TVM")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      *ret = hclir_to_tvm(args[0], args[1]);
    });

}  // namespace hcl
