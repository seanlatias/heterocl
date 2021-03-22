/*!
 *  Copyright (c) 2021 by Contributors
 * \file base.h
 */

#ifndef HCL_AST_BASE_H_
#define HCL_AST_BASE_H_

#include "../../tvm/expr.h"

namespace hcl {
namespace ast {

using TVM::Node;
using TVM::NodeRef;
using TVM::AttrVisitor;
using TVM::runtime::TVMArgs;
using TVM::runtime::TVMRetValue;

class Location;

/* Location for an ast node */
class LocationNode : public Node {
 public:
  std::string file_name;
  int line;
  int column;

  static Location make(std::string file_name, int line, int column);

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("file_name", &file_name);
    v->Visit("line", &line);
    v->Visit("column", &column);
  }

  static constexpr const char* _type_key = "Location";
  TVM_DECLARE_NODE_TYPE_INFO(LocationNode, Node);
};

class Location : public NodeRef {
 public:
  Location() {}
  explicit Location(std::shared_ptr<Node> n) : NodeRef(n) {}

  inline const LocationNode* operator->() const;
  using ContainerType = LocationNode;
};

// implements of inline functions
inline const LocationNode* Location::operator->() const {
  return static_cast<const LocationNode*>(node_.get());
}

}  // namespace ast
}  // namespace hcl

#endif  // HCL_AST_BASE_H_
