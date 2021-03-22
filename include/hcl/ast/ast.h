/*!
 *  Copyright (c) 2021 by Contributors
 * \file ast.h
 */

#ifndef HCL_AST_AST_H_
#define HCL_AST_AST_H_

#include "node.h"

namespace hcl {
namespace ast {

class Var : public ASTExprNode<Var> {
 public:
   std::string name;

  static ASTExpr make(Location loc, ASTType type, std::string name);

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("loc", &loc);
    v->Visit("type", &type);
    v->Visit("name", &name);
  }

  static constexpr const char* _type_key = "ASTVar";
};

class Add : public ASTExprNode<Add> {
 public:
  ASTExpr lhs;
  ASTExpr rhs;

  static ASTExpr make(Location loc, ASTType type, ASTExpr lhs, ASTExpr rhs);

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("loc", &loc);
    v->Visit("type", &type);
    v->Visit("lhs", &lhs);
    v->Visit("rhs", &rhs);
  }

  static constexpr const char* _type_key = "ASTAdd";
};

}  // namespace ast
}  // namespace hcl

#endif  // HCL_AST_AST_H_
