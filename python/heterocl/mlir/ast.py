"""AST Node in HeteroCL"""
from ..tvm._ffi.node import NodeBase, register_node

@register_node("Location")
class Location(NodeBase):
    pass

@register_node("MLIRModule")
class MLIRModule(NodeBase):
    pass

class ASTType(NodeBase):
    pass

@register_node("ASTNone")
class ASTInt(ASTType):
    pass

@register_node("ASTInt")
class ASTInt(ASTType):
    pass

@register_node("ASTFloat")
class ASTFloat(ASTType):
    pass

@register_node("ASTTensorType")
class ASTTensorType(ASTType):
    pass

class ASTExpr(NodeBase):
    pass

class ASTStmt(NodeBase):
    pass

@register_node("ASTPlaceholder")
class ASTPlaceholder(ASTExpr):
    pass

@register_node("ASTVarDeclare")
class ASTVarDeclare(ASTExpr):
    pass

@register_node("ASTAdd")
class ASTAdd(ASTExpr):
    pass

@register_node("ASTCompute")
class ASTCompute(ASTStmt):
    pass

@register_node("ASTFunction")
class ASTFunction(ASTStmt):
    pass

@register_node("ASTModule")
class ASTModule(ASTStmt):
    pass

@register_node("ASTRegion")
class ASTRegion(ASTStmt):
    pass

@register_node("ASTBlock")
class ASTBlock(ASTStmt):
    pass

