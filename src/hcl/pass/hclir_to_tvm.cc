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

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"

using namespace TVM;
using namespace TVM::ir;
using namespace hcl::ast;

namespace {

class GenTVMIR {
 public:
  GenTVMIR(mlir::ModuleOp module) : module_(module) {}

  Stmt generate(Array<Buffer> extern_buffer) {
    auto& block = module_.getRegion().front();
    for (mlir::Operation &op : block.getOperations()) {
      op.getName().print(llvm::errs());
    }
    return Stmt();
  }

 private:
  mlir::ModuleOp module_;
  std::map <mlir::Value, Buffer> buffers_;
};

}  // namespace


namespace hcl {

Stmt hclir_to_tvm(MLIRModule mod, Array<Buffer> extern_buffer) {
  mlir::ModuleOp module = mod->module.get();
  module.dump();
  return GenTVMIR(module).generate(extern_buffer);
}

TVM_REGISTER_API("ir_pass.HCL2TVM")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      *ret = hclir_to_tvm(args[0], args[1]);
    });

}  // namespace hcl
