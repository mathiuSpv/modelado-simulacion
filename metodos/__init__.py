import pandas as pd
import numpy as np
import sympy as sp
import plotly.graph_objects as go

# 9 Decimales
pd.set_option('display.precision', 9)
pd.set_option('display.float_format', '{:.9f}'.format)

MODULES = [
    {
        'sqrt': np.sqrt,
        'exp': np.exp,
        'sin': np.sin,
        'cos': np.cos,
        'log': np.log,
        'e': np.e,
    },
    'numpy',
    'math'
]

TOLERANCIA = 1e-9
MAX_ITER = 100

__all__ = ["pd", "np", "sp", "go", "TOLERANCIA", "MAX_ITER", "MODULES"]