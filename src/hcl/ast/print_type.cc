/*!
 *  Copyright (c) 2021 by Contributors
 * \file print_type.cc
 */
#include <hcl/ast/type.h>

namespace hcl {
namespace ast {

using Halide::Internal::IRPrinter;

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
    .set_dispatch<ASTInt>([](const ASTInt *op, IRPrinter *p) {
      if (!op->is_signed) p->stream << "u";
      if (op->nbits == op->nints) {
        p->stream << "int" << op->nbits;
      } else {
        p->stream << "fixed" << op->nbits << "_" << op->nints;
      }
    });

}  // namespace ast
}  // namespace hcl
