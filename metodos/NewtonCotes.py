try:
    from . import pd, np, go, sp, TOLERANCIA, MAX_ITER
except ImportError:
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    import sympy as sp
    TOLERANCIA = 1e-9
    MAX_ITER = 100

from enum import Enum
from pydantic import BaseModel
from typing import Optional, Dict, Any, Tuple, Callable

class IntegrationMethod(Enum):
    HALF_RECTANGLE = "Rectangulo_Medio"
    SIMPLE_TRAPEZOID = "Trapecio_Simple"
    COMPOUND_TRAPEZOID = "Trapecio_Compuesto"
    SIMPSON_SIMPLE = "Simpson_Simple"
    SIMPSON_COMPOUND = "Simpson_Compuesto"
    SIMPSON_3_8_SIMPLE = "Simpson_3/8_Simple"
    SIMPSON_3_8_COMPOUND = "Simpson_3/8_Compuesto"

    def __str__(self):
        return self.value

class NewtonCotesRequest(BaseModel):
    function: str
    lower_bound: float
    upper_bound: float
    num_intervals: int
    method: IntegrationMethod

class NewtonCotesResponse(BaseModel):
    result: float
    method: str
    function: str
    plot_data: Dict[str, Any]
    real_result: Optional[float] = None
    error: Optional[str] = None

class NewtonCotesCalculator:
    def __init__(self, request: NewtonCotesRequest):
        self.a = request.lower_bound
        self.b = request.upper_bound
        self.n = max(1, request.num_intervals)
        self.method = request.method
        self.function, self.function_repr = self._setup_function(request.function)
        self._validate_inputs()

    def _validate_inputs(self):
        """Valida límites del intervalo y método seleccionado."""
        if self.a >= self.b:
            raise ValueError(f"Intervalo inválido: a ({self.a}) >= b ({self.b})")
        if self.method in [IntegrationMethod.SIMPSON_COMPOUND, IntegrationMethod.SIMPSON_3_8_COMPOUND] and self.n <= 1:
            raise ValueError(f"Método {self.method} requiere n > 1")

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

    def _half_rectangle(self) -> float:
        h = (self.b - self.a) / self.n
        return sum(self.function(self.a + i*h + h/2) * h for i in range(self.n))

    def _trapezoid_simple(self) -> float:
        h = (self.b - self.a) / self.n
        return sum((self.function(self.a + i*h) + self.function(self.a + (i+1)*h)) * h/2 for i in range(self.n))

    def _simpson_simple(self) -> float:
        h = (self.b - self.a) / 2
        return (self.function(self.a) + 4*self.function(self.a + h) + self.function(self.b)) * h/3

    def _generate_plot(self, x_vals: np.ndarray, y_vals: np.ndarray) -> Dict[str, Any]:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals, fill='tozeroy', 
            name=f'Área ({self.method})'))
        fig.update_layout(
            title=f'Integración Numérica ({self.method})',
            xaxis_title='x',
            yaxis_title='f(x)')
        return fig.to_dict()

    def execute(self) -> NewtonCotesResponse:
        method_map = {
            IntegrationMethod.HALF_RECTANGLE: self._half_rectangle,
            IntegrationMethod.SIMPLE_TRAPEZOID: self._trapezoid_simple,
            IntegrationMethod.SIMPSON_SIMPLE: self._simpson_simple
        }
        
        try:
            result = method_map[self.method]()
            x_vals = np.linspace(self.a, self.b, 100)
            y_vals = [self.function(x) for x in x_vals]
            
            real_result = float(sp.integrate(
                sp.sympify(self.function_repr), 
                (sp.symbols('x'), self.a, self.b)
            ))

            return NewtonCotesResponse(
                result=result,
                method=str(self.method),
                function=self.function_repr,
                plot_data=self._generate_plot(x_vals, y_vals),
                real_result=real_result
            )
        except Exception as e:
            return NewtonCotesResponse(
                result=0.0,
                method=str(self.method),
                function=self.function_repr,
                plot_data={},
                error=str(e)
            )

    def __str__(self):
        return str(self.execute())