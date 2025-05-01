try:
    from . import pd, np, go, sp
except ImportError:
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    import sympy as sp

from pydantic import BaseModel, field_validator
from typing import Dict, Any, Tuple, Callable

class MontecarloRequest(BaseModel):
    funcion: str
    a: float
    b: float
    n: int
    seed: int = 0

    @field_validator('b')
    def validate_interval(cls, v: float, values: Any):
        """El intervalo [a,b] debe ser válido (a < b)"""
        if 'a' in values.data and v <= values.data['a']:
            raise ValueError("b debe ser mayor que a")
        return v

    @field_validator('n')
    def validate_samples(cls, v: float):
        """El número de muestras debe ser positivo"""
        if v <= 0:
            raise ValueError("n debe ser positivo")
        return v

class MontecarloResponse(BaseModel):
    media: float
    desviacion_estandar: float
    funcion: str
    plot_data: Dict[str, Any]

class MontecarloCalculator:
    def __init__(self, request: MontecarloRequest):
        self.a = request.a
        self.b = request.b
        self.n = max(1, request.n)
        self.seed = np.random.seed(request.seed)
        self.function, self.function_repr = self._setup_function(request.funcion)

    def _setup_function(self, func_str: str) -> Tuple[Callable, str]:
        x = sp.symbols('x')
        try:
            expr = sp.sympify(func_str)
            return (
                sp.lambdify(x, expr, modules=['numpy']),
                str(expr)
            )
        except sp.SympifyError:
            raise ValueError(f"Función inválida: {func_str}")

    def _generate_plot(self, x_aleatorios: np.ndarray, y_eval: np.ndarray) -> Dict[str, Any]:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=y_eval,
            name='Distribución',
            marker_color='#1f77b4',
            opacity=0.75,
            nbinsx=50
        ))
        fig.update_layout(
            title=f'Monte Carlo (n={self.n})',
            xaxis_title='f(x)',
            yaxis_title='Frecuencia',
            bargap=0.05
        )
        return fig.to_dict()
    
    def toDataFrame(self) -> pd.DataFrame:
        """Devuelve los resultados estadísticos como DataFrame con una fila:
        [media, desviacion_estandar]"""
        result = self.execute()
        return pd.DataFrame([result.model_dump(exclude={'funcion', 'plot_data'})])

    def execute(self) -> MontecarloResponse:
        if self.seed is not None:
            np.random.seed(self.seed)

        x_aleatorios = np.random.uniform(self.a, self.b, self.n)
        y_eval = self.function(x_aleatorios)

        media = np.mean(y_eval)
        desviacion = np.std(y_eval, ddof=1)

        return MontecarloResponse(
            media=media,
            desviacion_estandar=desviacion,
            funcion=self.function_repr,
            plot_data=self._generate_plot(x_aleatorios, y_eval)
        )

    def __str__(self):
        return str(self.execute())