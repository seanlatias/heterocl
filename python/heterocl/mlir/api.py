import inspect
from ..tvm import make as _make
from .util import get_location
from .type import NoneType, TensorType

def placeholder(shape, name, dtype):
    frame = inspect.stack()[1]
    loc = get_location(frame, "placeholder")
    return _make.ASTPlaceholder(loc, TensorType(dtype, shape).get(), name)

def var(shape, name, dtype):
    frame = inspect.stack()[1]
    loc = get_location(frame, "var")
    return _make.ASTVarDeclare(loc, TensorType(dtype, shape).get(), name)

def add(lhs, rhs):
    frame = inspect.stack()[1]
    loc = get_location(frame, "add")
    return _make.ASTAdd(loc, lhs.type, lhs, rhs)

def compute(dest, expr):
    frame = inspect.stack()[1]
    loc = get_location(frame, "compute")
    return _make.ASTCompute(loc, dest, expr)

def create_schedule(args, body, name):
    frame = inspect.stack()[1]
    loc = get_location(frame, "create_schedule")
    func = _make.ASTFunction(loc, name, args, NoneType().get(), body)
    return _make.ASTModule(loc, name, [func])
