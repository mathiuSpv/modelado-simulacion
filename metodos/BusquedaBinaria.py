try:
    from . import pd, np, go, sp, TOLERANCIA, MAX_ITER, MODULES
except ImportError:
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    import sympy as sp
    TOLERANCIA = 1e-9
    MAX_ITER = 100

from pydantic import BaseModel, field_validator
from typing import List, Dict, Any, Tuple, Callable

class BusquedaBinariaRequest(BaseModel):
    function: str
    a: float
    b: float
    tolerance: float = None
    max_iterations: int = None

    @field_validator('b')
    def validate_interval(cls, v: float, value: Any):
        """Valida que a < b"""
        if 'a' not in value.data:
            return v
        a = value.data['a']
        if v <= a:
            raise ValueError("b debe ser mayor que a")
        return v
    
class BusquedaBinariaRowResponse(BaseModel):
    iteration: int
    a: float
    b: float
    midpoint: float
    f_midpoint: float

class BusquedaBinariaResponse(BaseModel):
    root: float
    iterations: List[BusquedaBinariaRowResponse]
    function: str
    plot_data: Dict[str, Any]

class BusquedaBinariaCalculator:
    def __init__(self, request: BusquedaBinariaRequest):
        self.a = request.a
        self.b = request.b
        self.tolerance = abs(request.tolerance) if request.tolerance else TOLERANCIA
        self.max_iter = max(1, request.max_iterations or MAX_ITER)
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

    def _generate_plot(self, iterations: List[BusquedaBinariaRowResponse]) -> Dict[str, Any]:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[i.iteration for i in iterations],
            y=[i.midpoint for i in iterations],
            mode='lines+markers',
            name='Convergencia',
            line=dict(color='firebrick', width=2)
        ))
        fig.update_layout(
            title='Búsqueda Binaria',
            xaxis_title='Iteración',
            yaxis_title='Punto Medio'
        )
        return fig.to_dict()
    
    def toDataFrame(self) -> pd.DataFrame:
        """Devuelve los resultados como DataFrame con columnas:
        [iteration, a, b, midpoint, f_midpoint]"""
        result = self.execute()
        data = [row.model_dump() for row in result.iterations]
        return pd.DataFrame(data).to_string(index=False)

    def execute(self) -> BusquedaBinariaResponse:
        iterations = []
        a, b = self.a, self.b

        for i in range(self.max_iter):
            midpoint = (a + b) / 2
            f_mid = self.function(midpoint)
            
            iterations.append(BusquedaBinariaRowResponse(
                iteration=i + 1,
                a=a,
                b=b,
                midpoint=midpoint,
                f_midpoint=f_mid
            ))

            if abs(f_mid) < self.tolerance or (b - a)/2 < self.tolerance:
                break

            if self.function(a) * f_mid < 0:
                b = midpoint
            else:
                a = midpoint

        return BusquedaBinariaResponse(
            root=midpoint,
            iterations=iterations,
            function=self.function_repr,
            plot_data=self._generate_plot(iterations)
        )

    def __str__(self):
        return str(self.execute())

if __name__ == "__main__":
    import __config__
    from metodos.test.pdf1BusquedaBinaria import testCases
    
    testCases()