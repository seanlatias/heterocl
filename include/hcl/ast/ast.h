/*!
 *  Copyright (c) 2021 by Contributors
 * \file ast.h
 */

#ifndef HCL_AST_AST_H_
#define HCL_AST_AST_H_

#include "node.h"

namespace hcl {
namespace ast {

/* The inputs to a module */
class Placeholder : public ASTExprNode<Placeholder> {
 public:
  std::string name;

  static ASTExpr make(Location loc, ASTType type, std::string name);

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("loc", &loc);
    v->Visit("type", &type);
    v->Visit("name", &name);
  }

  static constexpr const char* _type_key = "ASTPlaceholder";
};

/* Declaration of internal variables */
class VarDeclare : public ASTExprNode<VarDeclare> {
 public:
  std::string name;

  static ASTExpr make(Location loc, ASTType type, std::string name);

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("loc", &loc);
    v->Visit("type", &type);
    v->Visit("name", &name);
  }

  static constexpr const char* _type_key = "ASTVarDeclare";
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

/* ---------------------------------------------------------- */
/* Operations                                                 */
/* ---------------------------------------------------------- */

/* A computation that assigns expr to dest */
class Compute : public ASTStmtNode<Compute> {
 public:
  ASTExpr dest;
  ASTExpr expr;

  static ASTStmt make(Location loc, ASTExpr dest, ASTExpr expr);

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("loc", &loc);
    v->Visit("dest", &dest);
    v->Visit("expr", &expr);
  }

  static constexpr const char* _type_key = "ASTCompute";
};

/* The top-level operation */
class Module : public ASTStmtNode<Module> {
 public:
  Array<ASTStmt> regions;

  static ASTStmt make(Location loc, Array<ASTStmt> regions);

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("loc", &loc);
    v->Visit("regions", &regions);
  }

  static constexpr const char* _type_key = "ASTModule";
};

/* Regions can be computed in parallel */
class Region : public ASTStmtNode<Region> {
 public:
  Array<ASTStmt> blocks;

  static ASTStmt make(Location loc, Array<ASTStmt> blocks);

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("loc", &loc);
    v->Visit("blocks", &blocks);
  }

  static constexpr const char* _type_key = "ASTRegion";
};

/* Blocks can be computed in sequential or parallel */
class Block : public ASTStmtNode<Block> {
 public:
  Array<ASTStmt> operations;

  static ASTStmt make(Location loc, Array<ASTStmt> operations);

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("loc", &loc);
    v->Visit("operations", &operations);
  }

  static constexpr const char* _type_key = "ASTBlock";
};

}  // namespace ast
}  // namespace hcl

#endif  // HCL_AST_AST_H_
