"""AST Node in HeteroCL"""
from .tvm._ffi.node import NodeBase, register_node

@register_node("Location")
class Location(NodeBase):
    pass

class ASTType(NodeBase):
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

@register_node("ASTVar")
class ASTVar(ASTExpr):
    pass

@register_node("ASTAdd")
class ASTAdd(ASTExpr):
    pass


