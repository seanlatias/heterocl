//===- HCLIROps.td -----------------------*- tablegen -*-===//
//  Copyright (c) 2021 by Contributors
//
//  Operations for HeteroCL IR
//===----------------------------------------------------===//

#ifndef HCL_DIALECT_HCLIR_OPS
#define HCL_DIALECT_HCLIR_OPS

include "HCLIR/HCLIRDialect.td"

def AllocateOp : HCLIR_Op<"allocate"> {

    let summary = "Allocate a variable";
    let description = [{
        Allcoate a variable
    }];

    let results = (outs AnyType);
}

def AddOp : HCLIR_Op<"add"> {

    let summary = "Add operator";
    let description = [{
        Add two operand
    }];

    let arguments = (ins AnyType:$a, AnyType:$b);
    let results = (outs AnyType);
}

def ComputeOp : HCLIR_Op<"compute"> {

    let summary = "Compute an expression";
    let description = [{
        Compute an expression
    }];

    let arguments = (ins AnyType:$dest, AnyType:$expr);
}

#endif // HCL_DIALECT_HCLIR_OPS
