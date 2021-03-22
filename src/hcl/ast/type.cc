/*!
 *  Copyright (c) 2021 by Contributors
 * \file type.cc
 */
#include <tvm/api_registry.h>
#include <hcl/ast/type.h>

namespace hcl {
namespace ast {

ASTType ASTInt::make(bool is_signed, uint64_t nbits, uint64_t nints) {
  std::shared_ptr<ASTInt> node = std::make_shared<ASTInt>();
  node->is_signed = is_signed;
  node->nbits = nbits;
  node->nints = nints;
  return ASTType(node);
}

TVM_REGISTER_NODE_TYPE(ASTInt);

TVM_REGISTER_API("make.ASTInt")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      if (args.size() == 2) {
        *ret = ASTInt::make(args[0], args[1], args[1]);
      } else {
        *ret = ASTInt::make(args[0], args[1], args[2]);
      }
    });

}  // namespace ast
}  // namespace hcl
