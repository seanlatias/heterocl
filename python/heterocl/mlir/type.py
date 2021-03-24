from ..tvm import make as _make

class Type():
    def get(self):
        pass

class Int(Type):
    def __init__(self, nbits=32, nints=32, signed=True):
        self.nbits = nbits
        self.nints = nints
        self.signed = signed

    def get(self):
        return _make.ASTInt(self.signed, self.nbits, self.nints)

class TensorType(Type):
    def __init__(self, dtype, shape):
        self.dtype = dtype
        self.shape = shape

    def get(self):
        return _make.ASTTensorType(self.dtype.get(), self.shape)
