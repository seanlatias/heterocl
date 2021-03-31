/*!
 *  Copyright (c) 2021 by Contributors
 * \file type.cc
 */
#include <tvm/api_registry.h>
#include <hcl/ast/type.h>

namespace hcl {
namespace ast {

ASTType ASTNone::make() {
  std::shared_ptr<ASTNone> node = std::make_shared<ASTNone>();
  return ASTType(node);
}

ASTType ASTInt::make(bool is_signed, uint64_t nbits, uint64_t nints) {
  std::shared_ptr<ASTInt> node = std::make_shared<ASTInt>();
  node->is_signed = is_signed;
  node->nbits = nbits;
  node->nints = nints;
  return ASTType(node);
}

ASTType ASTFloat::make(uint64_t nbits) {
  std::shared_ptr<ASTFloat> node = std::make_shared<ASTFloat>();
  node->nbits = nbits;
  node->nexps = nbits == 32 ? 8 : 11;
  node->nmants = nbits == 32 ? 23 : 52;
  return ASTType(node);
}

ASTType ASTFloat::make(uint64_t nbits, uint64_t nexps, uint64_t nmants) {
  std::shared_ptr<ASTFloat> node = std::make_shared<ASTFloat>();
  node->nbits = nbits;
  node->nexps = nexps;
  node->nmants = nmants;
  return ASTType(node);
}

ASTType ASTTensorType::make(ASTType type, Array<Expr> dims) {
  std::shared_ptr<ASTTensorType> node = std::make_shared<ASTTensorType>();
  node->type = std::move(type);
  node->dims = std::move(dims);
  return ASTType(node);
}

TVM_REGISTER_NODE_TYPE(ASTNone);
TVM_REGISTER_NODE_TYPE(ASTInt);
TVM_REGISTER_NODE_TYPE(ASTFloat);
TVM_REGISTER_NODE_TYPE(ASTTensorType);

TVM_REGISTER_API("make.ASTNone")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      *ret = ASTNone::make();
    });

TVM_REGISTER_API("make.ASTInt")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      if (args.size() == 2) {
        *ret = ASTInt::make(args[0], args[1], args[1]);
      } else {
        *ret = ASTInt::make(args[0], args[1], args[2]);
      }
    });

TVM_REGISTER_API("make.ASTFloat")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      if (args.size() == 1) {
        *ret = ASTFloat::make(args[0]);
      } else {
        *ret = ASTFloat::make(args[0], args[1], args[2]);
      }
    });

TVM_REGISTER_API("make.ASTTensorType")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      *ret = ASTTensorType::make(args[0], args[1]);
    });

}  // namespace ast
}  // namespace hcl
