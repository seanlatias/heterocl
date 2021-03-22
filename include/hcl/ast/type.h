/*!
 *  Copyright (c) 2021 by Contributors
 * \file base.h
 */

#ifndef HCL_AST_TYPE_H_
#define HCL_AST_TYPE_H_

#include "base.h"

namespace hcl {
namespace ast {

using TVM::Array;
using Halide::Expr;

class ASTTypeNode : public Node {
 public:
  static constexpr const char* _type_key = "ASTType";
  TVM_DECLARE_BASE_NODE_INFO(ASTTypeNode, Node);
};

class ASTType : public NodeRef {
 public:
  ASTType() {}
  explicit ASTType(std::shared_ptr<Node> n) : NodeRef(n) {}

  inline const ASTTypeNode* operator->() const;
  using ContainerType = ASTTypeNode;
};

class ASTInt : public ASTTypeNode {
 public:
  bool is_signed;
  uint64_t nbits;
  uint64_t nints;

  static ASTType make(bool is_signed, uint64_t nbits, uint64_t nints);

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("is_signed", &is_signed);
    v->Visit("nbits", &nbits);
    v->Visit("nints", &nints);
  }

  static constexpr const char* _type_key = "ASTInt";
  TVM_DECLARE_NODE_TYPE_INFO(ASTInt, ASTTypeNode);
};

class ASTFloat : public ASTTypeNode {
 public:
  uint64_t nbits;
  uint64_t nexps;   // exponent
  uint64_t nmants;  // mantissa

  static ASTType make(uint64_t nbits);
  static ASTType make(uint64_t nbits, uint64_t nexps, uint64_t nmants);

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("nbits", &nbits);
    v->Visit("nexps", &nexps);
    v->Visit("nmants", &nmants);
  }

  static constexpr const char* _type_key = "ASTFloat";
  TVM_DECLARE_NODE_TYPE_INFO(ASTFloat, ASTTypeNode);
};

class ASTTensorType : public ASTTypeNode {
 public:
  ASTType type;
  Array<Expr> dims;

  static ASTType make(ASTType type, Array<Expr> dims);

  size_t ndim() { return dims.size(); }

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("type", &type);
    v->Visit("dims", &dims);
  }

  static constexpr const char* _type_key = "ASTTensorType";
  TVM_DECLARE_NODE_TYPE_INFO(ASTTensorType, ASTTypeNode);
};

// implements of inline functions
inline const ASTTypeNode* ASTType::operator->() const {
  return static_cast<const ASTTypeNode*>(node_.get());
}

}  // namespace ast
}  // namespace hcl

#endif  // HCL_AST_TYPE_H_
