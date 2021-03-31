/*!
 *  Copyright (c) 2021 by Contributors
 * \file print_type.cc
 */
#include <hcl/ast/type.h>

namespace hcl {
namespace ast {

using Halide::Internal::IRPrinter;

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
    .set_dispatch<ASTNone>([](const ASTNone *op, IRPrinter *p) {
      p->stream << "void";
    });

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
    .set_dispatch<ASTInt>([](const ASTInt *op, IRPrinter *p) {
      if (!op->is_signed) p->stream << "u";
      if (op->nbits == op->nints) {
        p->stream << "int" << op->nbits;
      } else {
        p->stream << "fixed" << op->nbits << "_" << op->nints;
      }
    });

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
    .set_dispatch<ASTFloat>([](const ASTFloat *op, IRPrinter *p) {
      p->stream << "float" << op->nbits;
    });

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
    .set_dispatch<ASTTensorType>([](const ASTTensorType *op, IRPrinter *p) {
      p->stream << "tensor<";
      for (size_t i = 0; i < op->dims.size(); i++) {
        p->print(op->dims[i]);
        p->stream << " x ";
      }
      p->print(op->type);
      p->stream << ">";
    });


}  // namespace ast
}  // namespace hcl
