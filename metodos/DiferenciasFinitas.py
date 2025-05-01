try:
    from . import pd, np, go, sp, MODULES
except ImportError:
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    import sympy as sp

from pydantic import BaseModel, field_validator
from typing import Dict, Any, Tuple, Callable

class DiferenciasFinitasRequest(BaseModel):
    function
    x: float
    h: float

    @field_validator('x')
    def validate_x(cls, v: float):
        """El punto de evaluación debe ser finito"""
        if not np.isfinite(v):
            raise ValueError("x debe ser un número finito")
        return v

    @field_validator('h')
    def validate_h(cls, v):
        """El paso h debe ser positivo para aproximación numérica estable"""
        if v <= 0:
            raise ValueError("h debe ser positivo")
        return abs(v)

class DerivadaResponse(BaseModel):
    primera_derivada: float
    segunda_derivada: float
    function
    plot_data: Dict[str, Any]

class DiferenciasFinitasCalculator:
    def __init__(self, request: DiferenciasFinitasRequest):
        self.x = request.x
        self.h = abs(request.h)
        self.function, self.function_repr = self._setup_function(request.function)

    def _setup_function(self, func_str: str) -> Tuple[Callable, str]:
        x = sp.symbols('x')
        try:
            expr = sp.sympify(func_str)
            return (
                sp.lambdify(x, expr, modules=MODULES),
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
    
    def toDataFrame(self) -> pd.DataFrame:
        """Devuelve las derivadas como DataFrame con una sola fila:
        [primera_derivada, segunda_derivada]"""
        result = self.execute()
        return pd.DataFrame([result.model_dump(exclude={'function', 'plot_data'})]).to_string(index=False)

    def execute(self) -> DerivadaResponse:
        primera, segunda = self._calcular_derivadas()
        return DerivadaResponse(
            primera_derivada=primera,
            segunda_derivada=segunda,
            function=self.function_repr,
            plot_data=self._generate_plot()
        )

    def __str__(self):
        return str(self.execute())