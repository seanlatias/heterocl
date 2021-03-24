from ..tvm import make as _make

def get_location(frame, name):
    filename = frame.filename
    line_num = frame.lineno
    column_num = frame.code_context[0].find(name)
    return _make.Location(filename, line_num, column_num)
