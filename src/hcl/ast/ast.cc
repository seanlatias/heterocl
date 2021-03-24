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

ASTExpr Placeholder::make(Location loc, ASTType type, std::string name) {
  std::shared_ptr<Placeholder> node = std::make_shared<Placeholder>();
  node->loc = std::move(loc);
  node->type = std::move(type);
  node->name = name;
  return ASTExpr(node);
}

ASTExpr VarDeclare::make(Location loc, ASTType type, std::string name) {
  std::shared_ptr<VarDeclare> node = std::make_shared<VarDeclare>();
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

ASTStmt Compute::make(Location loc, ASTExpr dest, ASTExpr expr) {
  std::shared_ptr<Compute> node = std::make_shared<Compute>();
  node->loc = std::move(loc);
  node->dest = std::move(dest);
  node->expr = std::move(expr);
  return ASTStmt(node);
}

ASTStmt Module::make(Location loc, Array<ASTStmt> regions) {
  std::shared_ptr<Module> node = std::make_shared<Module>();
  node->loc = std::move(loc);
  node->regions = std::move(regions);
}

ASTStmt Region::make(Location loc, Array<ASTStmt> blocks) {
  std::shared_ptr<Region> node = std::make_shared<Region>();
  node->loc = std::move(loc);
  node->blocks = std::move(blocks);
}

ASTStmt Block::make(Location loc, Array<ASTStmt> operations) {
  std::shared_ptr<Block> node = std::make_shared<Block>();
  node->loc = std::move(loc);
  node->operations = std::move(operations);
}

TVM_REGISTER_NODE_TYPE(LocationNode);
TVM_REGISTER_NODE_TYPE(Placeholder);
TVM_REGISTER_NODE_TYPE(VarDeclare);
TVM_REGISTER_NODE_TYPE(Add);

TVM_REGISTER_NODE_TYPE(Compute);
TVM_REGISTER_NODE_TYPE(Module);
TVM_REGISTER_NODE_TYPE(Region);
TVM_REGISTER_NODE_TYPE(Block);

TVM_REGISTER_API("make.Location")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      *ret = LocationNode::make(args[0], args[1], args[2]);
    });

TVM_REGISTER_API("make.ASTPlaceholder")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      *ret = Placeholder::make(args[0], args[1], args[2]);
    });

TVM_REGISTER_API("make.ASTVarDeclare")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      *ret = VarDeclare::make(args[0], args[1], args[2]);
    });

TVM_REGISTER_API("make.ASTAdd")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      *ret = Add::make(args[0], args[1], args[2], args[3]);
    });

TVM_REGISTER_API("make.ASTCompute")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      *ret = Compute::make(args[0], args[1], args[2]);
    });

TVM_REGISTER_API("make.ASTModule")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      *ret = Module::make(args[0], args[1]);
    });

TVM_REGISTER_API("make.ASTRegion")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      *ret = Region::make(args[0], args[1]);
    });

TVM_REGISTER_API("make.ASTBlock")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      *ret = Block::make(args[0], args[1]);
    });

}  // namespace ast
}  // namespace hcl
