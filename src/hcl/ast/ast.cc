/*!
 *  Copyright (c) 2021 by Contributors
 * \file ast.cc
 */
#include <tvm/api_registry.h>
#include <hcl/ast/ast.h>

namespace hcl {
namespace ast {

Location LocationNode::make(std::string file_name, int line, int column) {
  std::shared_ptr<LocationNode> node = std::make_shared<LocationNode>();
  node->file_name = file_name;
  node->line = line;
  node->column = column;
  return Location(node);
}

ASTExpr Var::make(Location loc, ASTType type, std::string name) {
  std::shared_ptr<Var> node = std::make_shared<Var>();
  node->loc = std::move(loc);
  node->type = std::move(type);
  node->name = name;
  return ASTExpr(node);
}

ASTExpr Add::make(Location loc, ASTType type, ASTExpr lhs, ASTExpr rhs) {
  std::shared_ptr<Add> node = std::make_shared<Add>();
  node->loc = std::move(loc);
  node->type = std::move(type);
  node->lhs = std::move(lhs);
  node->rhs = std::move(rhs);
  return ASTExpr(node);
}

TVM_REGISTER_NODE_TYPE(LocationNode);
TVM_REGISTER_NODE_TYPE(Var);
TVM_REGISTER_NODE_TYPE(Add);

TVM_REGISTER_API("make.Location")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      *ret = LocationNode::make(args[0], args[1], args[2]);
    });

TVM_REGISTER_API("make.ASTVar")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      *ret = Var::make(args[0], args[1], args[2]);
    });

TVM_REGISTER_API("make.ASTAdd")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      *ret = Add::make(args[0], args[1], args[2], args[3]);
    });

}  // namespace ast
}  // namespace hcl
