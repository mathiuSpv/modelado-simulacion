import pandas as pd
import numpy as np
import sympy as sp
import plotly.graph_objects as go

# 9 Decimales
pd.set_option('display.precision', 9)
pd.set_option('display.float_format', '{:.9f}'.format)

TOLERANCIA = 1e-9
MAX_ITER = 100

__all__ = ["pd", "np", "sp", "go", "TOLERANCIA", "MAX_ITER"]