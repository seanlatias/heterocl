/*!
 *  Copyright (c) 2021 by Contributors
 * \file base.h
 */

#ifndef HCL_AST_NODE_H_
#define HCL_AST_NODE_H_

#include "type.h"

namespace hcl {
namespace ast {

/* The base class for all  AST nodes */
class ASTNode : public Node {
 public:
  Location loc;
};

class BaseASTExprNode : public ASTNode {
 public:
  ASTType type;

  static constexpr const char* _type_key = "ASTExpr";
  TVM_DECLARE_BASE_NODE_INFO(BaseASTExprNode, Node);
};

class BaseASTStmtNode : public ASTNode {
 public:
  static constexpr const char* _type_key = "ASTStmt";
  TVM_DECLARE_BASE_NODE_INFO(BaseASTStmtNode, Node);
};

class ASTExpr : public NodeRef {
 public:
  ASTExpr() {}
  explicit ASTExpr(std::shared_ptr<Node> n) : NodeRef(n) {}

  inline const BaseASTExprNode* operator->() const;
  using ContainerType = BaseASTExprNode;
};

class ASTStmt : public NodeRef {
 public:
  ASTStmt() {}
  explicit ASTStmt(std::shared_ptr<Node> n) : NodeRef(n) {}

  inline const BaseASTStmtNode* operator->() const;
  using ContainerType = BaseASTStmtNode;
};

template <typename T>
class ASTExprNode : public BaseASTExprNode {
 public:
  TVM_DECLARE_NODE_TYPE_INFO(T, BaseASTExprNode);
};

template <typename T>
class ASTStmtNode : public BaseASTStmtNode {
 public:
  TVM_DECLARE_NODE_TYPE_INFO(T, BaseASTStmtNode);
};

// implements of inline functions
inline const BaseASTExprNode* ASTExpr::operator->() const {
  return static_cast<const BaseASTExprNode*>(node_.get());
}

inline const BaseASTStmtNode* ASTStmt::operator->() const {
  return static_cast<const BaseASTStmtNode*>(node_.get());
}

}  // namespace ast
}  // namespace hcl

#endif  // HCL_AST_NODE_H_
