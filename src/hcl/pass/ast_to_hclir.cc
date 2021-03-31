/*!
 *  Copyright (c) 2021 by Contributors
 * \file ast_to_hclir.cc
 */

#include "hcl/ast/ast.h"
#include "hcl/dialect/HCLIR/HCLIRDialect.h"
#include "hcl/dialect/HCLIR/HCLIROps.h"
#include "hcl/dialect/context.h"
#include "hcl/pass.h"
#include "tvm/api_registry.h"
#include "tvm/expr.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

using namespace hcl::ast;
using namespace mlir::hclir;

namespace {

class GenHCLIRDialect {
 public:
  GenHCLIRDialect(mlir::MLIRContext* context) : builder_(context) {}

  mlir::ModuleOp generate(ASTStmt module) {
    if (const Module* op = module.as<Module>()) {
      module_ = mlir::ModuleOp::create(loc(op->loc));

      for (size_t i = 0; i < op->functions.size(); i++) {
        module_.push_back(gen_func(op->functions[i].as<Function>()));
      }

      return module_;
    }
    return nullptr;
  }

 private:
  mlir::ModuleOp module_;
  mlir::OpBuilder builder_;
  std::map<std::string, mlir::Value> symbols_;

  mlir::Location loc(const Location& loc) {
    return mlir::FileLineColLoc::get(builder_.getIdentifier(loc->file_name),
                                     loc->line, loc->column);
  }

  mlir::Type get_type(const ASTType& type) {
    if (type.as<ASTNone>()) {
      return builder_.getNoneType();
    } else if (const ASTInt* op = type.as<ASTInt>()) {
      return builder_.getIntegerType(op->nbits, op->is_signed);
    } else if (const ASTFloat* op = type.as<ASTFloat>()) {
      return op->nbits == 32 ? builder_.getF32Type() : builder_.getF64Type();
    } else if (const ASTTensorType* op = type.as<ASTTensorType>()){
      // extract dims
      std::vector<int64_t> shape;
      for (size_t i = 0; i < op->dims.size(); i++) {
        const int64_t* val = as_const_int(op->dims[i]);
        shape.push_back(*val);
      }
      return mlir::RankedTensorType::get(shape, get_type(op->type));
    }
    return builder_.getNoneType();
  }

  void declare_var(ASTExpr expr, mlir::Value val) {
    if (const Placeholder* op = expr.as<Placeholder>()) {
      symbols_[op->name] = val;
    }
  }

  void gen_ir(const ASTStmt& stmt) {
    if (const Compute* op = stmt.as<Compute>()) {
      gen_ir(op);
    }
  }

  mlir::Value gen_ir(const ASTExpr& expr) {
    if (const VarDeclare* op = expr.as<VarDeclare>()) {
      return gen_ir(op);
    } else if (const Placeholder* op = expr.as<Placeholder>()) {
      return symbols_[op->name];
    } else if (const Add* op = expr.as<Add>()) {
      return gen_ir(op);
    }
    return nullptr;
  }

  mlir::FuncOp gen_func(const Function* op) {
    // generate prototype
    std::vector<mlir::Type> arg_types;
    for (size_t i = 0; i < op->args.size(); i++) {
      arg_types.push_back(get_type(op->args[i].type()));
    }
    mlir::Type ret_type = get_type(op->ret_type);
    auto func_type = builder_.getFunctionType(arg_types, ret_type);
    mlir::FuncOp func = mlir::FuncOp::create(loc(op->loc), op->name, func_type);

    // preparing the entry block
    mlir::Block* entry_block = func.addEntryBlock();
    auto block_args = entry_block->getArguments();
    for (size_t i = 0; i < op->args.size(); i++) {
      declare_var(op->args[i], block_args[i]);
    }

    builder_.setInsertionPointToStart(entry_block);

    gen_ir(op->body);

    return func;
  }

  mlir::Value gen_ir(const VarDeclare* op) {
   if (symbols_.find(op->name) == symbols_.end()) {
      auto alloc = builder_.create<AllocateOp>(loc(op->loc), get_type(op->type));
      symbols_[op->name] = alloc;
      return alloc;
    } else {
      return symbols_[op->name];
    }
  }

  mlir::Value gen_ir(const Add* op) {
    return builder_.create<AddOp>(loc(op->loc), get_type(op->type),
                                  gen_ir(op->lhs), gen_ir(op->rhs));
  }

  void gen_ir(const Compute* op) {
    builder_.create<ComputeOp>(loc(op->loc), mlir::TypeRange(),
                               gen_ir(op->dest), gen_ir(op->expr));
  }
};

}  // namespace

namespace hcl {

MLIRModule ast_to_hclir(ASTStmt module) {
  mlir::MLIRContext* context = IRContext::context;
  context->getOrLoadDialect<mlir::hclir::HCLIRDialect>();
  auto m = GenHCLIRDialect(context).generate(module);
  auto n = MLIRModuleNode::make(m);
  n->module.get().dump();
  return n;
}

TVM_REGISTER_API("ir_pass.AST2HCL")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      *ret = ast_to_hclir(args[0]);
    });

}  // namespace hcl




