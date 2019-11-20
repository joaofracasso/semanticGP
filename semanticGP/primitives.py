import operator
import math
import numpy as np

def add(a, b):
    try: return operator.add(a, b)
    except (OverflowError, ValueError): return 1.0


def sub(a, b):
    try: return operator.sub(a, b)
    except (OverflowError, ValueError): return 1.0

def mul(a, b):
    try:
        result =  operator.mul(a, b)
        if abs(result) > 1e43:
            return 1.0
        else:
            return result
    except (OverflowError, ValueError): return 1.0


def div(left, right):
    if (abs(right) > 1e-43):
        result = left / right
        return result 
    else:
        return 1.0

def pow(x, y):
    try: 
        result =  math.pow(x, y)
        if abs(result) > 1e43:
            return 1.0
        else:
            return result 
    except (OverflowError, ValueError): return 1.0

def exp(value):
    if value < 1e2:
        return math.exp(value)
    else:
        return 1.0


def log(value):
    if (value > 1e-100):
        return math.log(value)
    else:
        return 1.0

def sin(x):
    try: return math.sin(x)
    except (OverflowError, ValueError): return 1.0

def cos(x):
    try: return math.cos(x)
    except (OverflowError, ValueError): return 1.0
