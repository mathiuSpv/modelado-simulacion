from functools import wraps
from pandas import DataFrame

def paramsHandlerAitken(func):
    """
    Decorador para verificar 
    """
    @wraps(func)
    def wrapper(df: DataFrame, *args, **kwargs):
        # Verificar que el DataFrame tenga exactamente 3 columnas
        if len(df.columns) != 3:
            raise ValueError("El DataFrame debe tener exactamente 3 columnas.")
        
        # Verificar que las columnas se llamen x_n0, x_n1, x_n2
        columnas_esperadas = {'x_n0', 'x_n1', 'x_n2'}
        if not columnas_esperadas.issubset(df.columns):
            raise ValueError("El DataFrame debe contener las columnas 'x_n0', 'x_n1' y 'x_n2'.")
        return func(df, *args, **kwargs)
    return wrapper