try:
    from . import pd, np, go, sp, TOLERANCIA, MAX_ITER, MODULES
except ImportError:
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    import sympy as sp
    TOLERANCIA = 1e-9
    MAX_ITER = 100

from pydantic import BaseModel, field_validator, FieldValidationInfo
from typing import List, Dict, Any, Callable

class PuntoFijoRequest(BaseModel):
    function: str
    x0: float
    tolerance: float = None
    max_iterations: int = None

    @field_validator('function')
    def validate_function_convergence(cls, v: str, info: FieldValidationInfo):
        """
        Verifica que |g'(x0)| < 1 (Condición de convergencia del método).
        """
        if not hasattr(info, 'data') or 'x0' not in info.data:
            return v
            
        x = sp.symbols('x')
        try:
            expr = sp.sympify(v)
            g_prime = sp.diff(expr, x)
            g_prime_x0 = g_prime.subs(x, info.data['x0'])
            
            if abs(float(g_prime_x0)) >= 1:
                raise ValueError(
                    f"|g'({info.data['x0']})| = {abs(float(g_prime_x0)):.2f} ≥ 1\n"
                    "Requisito: |g'(x0)| < 1 para convergencia"
                )
        except sp.SympifyError:
            raise ValueError(f"Función inválida: {v}")
        return v
    
    @field_validator('x0')
    def validate_x0(cls, v):
        """Verifica que x0 sea finito."""
        if not np.isfinite(v):
            raise ValueError("x0 debe ser un número finito")
        return v

    @field_validator('tolerance')
    def validate_tolerance(cls, v):
        """La tolerancia debe ser positiva."""
        if v <= 0:
            raise ValueError("La tolerancia debe ser > 0")
        return abs(v)

class PuntoFijoRowResponse(BaseModel):
    iteration: int
    xn: float
    g_xn: float

class PuntoFijoResponse(BaseModel):
    result: List[PuntoFijoRowResponse]
    function: str
    plot_data: Dict[str, Any]

class PuntoFijoCalculator:
    def __init__(self, request: PuntoFijoRequest):
        self.x0 = request.x0
        self.tolerance = abs(request.tolerance) if request.tolerance else TOLERANCIA
        self.max_iter = max(1, request.max_iterations or MAX_ITER)
        self.function, self.function_repr = self._setup_function(request.function)

    def _setup_function(self, func_str: str) -> tuple[Callable, str]:
        x = sp.symbols('x')
        expr = sp.sympify(func_str)
        return (
            sp.lambdify(x, expr, modules=MODULES),
            str(expr)
        )

    def _generate_plot(self, iterations: List[int], values: List[float]) -> Dict[str, Any]:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=iterations,
            y=values,
            mode='lines+markers',
            name='Convergencia Punto Fijo'
        ))
        fig.update_layout(
            title='Método de Punto Fijo',
            xaxis_title='Iteración',
            yaxis_title='Valor de x'
        )
        return fig.to_dict()
    
    def toDataFrame(self) -> pd.DataFrame:
        """Devuelve los resultados como DataFrame con columnas:
        [iteration, xn, g_xn]"""
        result = self.execute()
        data = [row.model_dump() for row in result.result]
        return pd.DataFrame(data).to_string(index=False)

    def execute(self) -> PuntoFijoResponse:
        results = []
        current_x = self.x0

        for i in range(self.max_iter):
            xn1 = float(self.function(current_x))
            
            results.append(PuntoFijoRowResponse(
                iteration=i,
                xn=current_x,
                g_xn=xn1
            ))
            
            if abs(xn1 - current_x) < self.tolerance:
                break
                
            current_x = xn1
        else:
            raise RuntimeError(
                f"No convergencia en {self.max_iter} iteraciones. "
                f"Último error: {abs(xn1 - current_x):.3e}"
            )

        return PuntoFijoResponse(
            result=results,
            function=self.function_repr,
            plot_data=self._generate_plot(
                iterations=[r.iteration for r in results],
                values=[r.g_xn for r in results]
            )
        )

    def __str__(self):
        return str(self.execute())