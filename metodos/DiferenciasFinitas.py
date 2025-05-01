try:
    from . import pd, np, go, sp
except ImportError:
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    import sympy as sp

from pydantic import BaseModel
from typing import Dict, Any, Tuple, Callable

class DiferenciasFinitasRequest(BaseModel):
    funcion: str
    x: float
    h: float

class DerivadaResponse(BaseModel):
    primera_derivada: float
    segunda_derivada: float
    funcion: str
    plot_data: Dict[str, Any]

class DiferenciasFinitasCalculator:
    def __init__(self, request: DiferenciasFinitasRequest):
        self.x = request.x
        self.h = abs(request.h)
        self.function, self.function_repr = self._setup_function(request.funcion)
        self._validate_inputs()

    def _validate_inputs(self):
        """Valida que x sea finito y h > 0."""
        if not np.isfinite(self.x):
            raise ValueError(f"x debe ser finito. Se recibió: {self.x}")
        if self.h <= 0:
            raise ValueError(f"h debe ser positivo. Se recibió: {self.h}")

    def _setup_function(self, func_str: str) -> Tuple[Callable, str]:
        x = sp.symbols('x')
        try:
            expr = sp.sympify(func_str)
            return (
                sp.lambdify(x, expr, modules=['numpy', 'math']),
                str(expr)
            )
        except sp.SympifyError:
            raise ValueError(f"Función inválida: {func_str}")

    def _calcular_derivadas(self) -> Tuple[float, float]:
        """Calcula derivadas numéricas."""
        f, h, x = self.function, self.h, self.x
        primera = (f(x + h) - f(x - h)) / (2 * h)
        segunda = (f(x + h) - 2*f(x) + f(x - h)) / (h**2)
        return primera, segunda

    def _generate_plot(self) -> Dict[str, Any]:
        x_vals = np.linspace(self.x - 3*self.h, self.x + 3*self.h, 200)
        y_vals = [self.function(xi) for xi in x_vals]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals, mode='lines', name='Función',
            line=dict(color='royalblue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=[self.x], y=[self.function(self.x)], 
            mode='markers', name='Punto de evaluación',
            marker=dict(color='firebrick', size=10)
        ))
        fig.update_layout(
            title=f'Derivadas en x = {self.x:.2f}',
            xaxis_title='x',
            yaxis_title='f(x)',
            hovermode='x unified'
        )
        return fig.to_dict()

    def execute(self) -> DerivadaResponse:
        primera, segunda = self._calcular_derivadas()
        return DerivadaResponse(
            primera_derivada=primera,
            segunda_derivada=segunda,
            funcion=self.function_repr,
            plot_data=self._generate_plot()
        )

    def __str__(self):
        return str(self.execute())