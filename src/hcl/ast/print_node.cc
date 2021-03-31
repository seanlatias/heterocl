/*!
 *  Copyright (c) 2021 by Contributors
 * \file print_node.cc
 */
#include <hcl/ast/ast.h>

namespace hcl {
namespace ast {

using Halide::Internal::IRPrinter;

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
    .set_dispatch<LocationNode>([](const LocationNode *op, IRPrinter *p) {
      p->stream << "(" << op->file_name << ": "
                << op->line << ":"
                << op->column << ")";
    });

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
    .set_dispatch<MLIRModuleNode>([](const MLIRModuleNode *op, IRPrinter *p) {
      op->module.get().dump();
    });

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
    .set_dispatch<Placeholder>([](const Placeholder *op, IRPrinter *p) {
      p->do_indent();
      p->stream << "Placeholder: ";
      p->stream << op->name << " ";
      p->print(op->type);
      p->print(op->loc);
    });

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
    .set_dispatch<VarDeclare>([](const VarDeclare *op, IRPrinter *p) {
      p->do_indent();
      p->stream << "VarDeclare: ";
      p->stream << op->name;
      p->print(op->type);
      p->print(op->loc);
    });

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
    .set_dispatch<Add>([](const Add *op, IRPrinter *p) {
      p->print(op->lhs);
      p->stream << "+";
      p->print(op->rhs);
    });

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
    .set_dispatch<Compute>([](const Compute *op, IRPrinter *p) {
      p->do_indent();
      p->print(op->dest);
      p->stream << " = ";
      p->print(op->expr);
      p->stream << "\n";
    });

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
    .set_dispatch<Function>([](const Function *op, IRPrinter *p) {
      p->do_indent();
      p->print(op->ret_type);
      p->stream << " " << op->name << "(";
      for (size_t i = 0; i < op->args.size(); i++) {
        p->print(op->args[i].type());
        p->stream << " ";
        p->print(op->args[i]);
        if (i != op->args.size()-1) p->stream << ", ";
      }
      p->stream << ") {\n";
      p->indent += 2;
      p->print(op->body);
      p->indent -= 2;
      p->do_indent();
      p->stream << "}\n";
    });

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
    .set_dispatch<Module>([](const Module *op, IRPrinter *p) {
      p->stream << "Module " << op->name << " { ";
      p->print(op->loc);
      p->stream << "\n";
      p->indent += 2;
      for (size_t i = 0; i < op->functions.size(); i++) {
        p->print(op->functions[i]);
      }
      p->indent -= 2;
      p->stream << "}\n";
    });

 TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
    .set_dispatch<Region>([](const Region *op, IRPrinter *p) {
      p->do_indent();
      p->stream << "Region { ";
      p->print(op->loc);
      p->stream << "\n";
      p->indent += 2;
      for (size_t i = 0; i < op->blocks.size(); i++) {
        p->print(op->blocks[i]);
      }
      p->indent -= 2;
      p->do_indent();
      p->stream << "}\n";
    });

 TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
    .set_dispatch<Block>([](const Block *op, IRPrinter *p) {
      p->do_indent();
      p->stream << "Block { ";
      p->print(op->loc);
      p->stream << "\n";
      p->indent += 2;
      for (size_t i = 0; i < op->operations.size(); i++) {
        p->print(op->operations[i]);
      }
      p->indent -= 2;
      p->do_indent();
      p->stream << "}\n";
    });

}  // namespace ast
}  // namespace hcl
