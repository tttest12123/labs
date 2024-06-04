import numpy as np
import operator

allowed_functions = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "add": operator.add,
    "sub": operator.sub,
    "pow": operator.pow
}

def safe_eval(expr, allowed_functions, **kwargs):
    try:
        result = eval(expr, {"__builtins__": None}, {**allowed_functions, **kwargs})
        return result
    except Exception as e:
        return str(e)