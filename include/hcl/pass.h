/*!
 *  Copyright (c) 2021 by Contributors
 * \file ast.h
 */

#ifndef HCL_PASS_H_
#define HCL_PASS_H_

#include "ast/ast.h"

namespace hcl {

using namespace ast;

void ast_to_hclir(ASTStmt module);

}  // namespace hcl

#endif  // HCL_PASS_H_

