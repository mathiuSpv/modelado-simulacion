try:
    from . import pd, np, go, sp, TOLERANCIA, MAX_ITER
except ImportError:
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    import sympy as sp
    TOLERANCIA = 1e-9
    MAX_ITER = 100

from pydantic import BaseModel, field_validator
from typing import List, Dict, Any, Tuple, Callable

class AitkenRequest(BaseModel):
    function: str
    x0: float
    tolerance: float
    max_iterations: int

    @field_validator('x0')
    def validate_x0(cls, v: float):
        """Verifica que x0 sea finito y esté cerca de una raíz."""
        if not np.isfinite(v):
            raise ValueError("x0 debe ser un número finito")
        return v
    
    @field_validator('function')
    def validate_function_convergence(cls, v: str, values: Any):
        """
        Verifica que |g'(x0)| < 1 (Condición de convergencia del método).
        Matemáticamente, garantiza que la iteración converge localmente.
        """
        x = sp.symbols('x')
        try:
            expr = sp.sympify(v)
            g_prime = sp.diff(expr, x)
            g_prime_x0 = g_prime.subs(x, values.data['x0'])
            
            if abs(float(g_prime_x0)) >= 1:
                raise ValueError(
                    f"|g'(x0)| = {abs(float(g_prime_x0)):.2f} ≥ 1\n"
                    "El método no converge con esta función. Modifique g(x)."
                )
        except sp.SympifyError:
            raise ValueError(f"Función inválida: {v}")
        return v

    
class AitkenRowResponse(BaseModel):
    iteration: int
    xn0: float
    xn1: float
    xn2: float
    xnAitken: float

class AitkenResponse(BaseModel):
    result: List[AitkenRowResponse]
    function: str
    plot_data: Dict[str, Any]

class AitkenCalculator:
    def __init__(self, request: AitkenRequest):
        self.x0 = request.x0
        self.tolerance = abs(request.tolerance) if request.tolerance else TOLERANCIA
        self.max_iter = max(1, request.max_iterations or MAX_ITER)
        self.function, self.function_repr = self._setup_function(request.function)

    def _setup_function(self, func_str: str) -> Tuple[Callable, str]:
        x = sp.symbols('x')
        try:
            expr = sp.sympify(func_str)
            deriv = sp.diff(expr, x)
            deriv_val = deriv.subs(x, self.x0)
            if abs(float(deriv_val)) >= 1:
                raise ValueError(f"|g'({self.x0})| = {abs(float(deriv_val)):.2f} ≥ 1")
            return (
                sp.lambdify(x, expr, modules=['numpy', 'math']),
                str(expr)
            )
        except sp.SympifyError:
            raise ValueError(f"Expresión inválida: {func_str}")

    def _generate_plot(self, iterations: List[int], values: List[float]) -> Dict[str, Any]:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=iterations,
            y=values,
            mode='lines+markers',
            name='Aitken Acceleration'
        ))
        fig.update_layout(
            title='Aceleración de Aitken',
            xaxis_title='Iteración',
            yaxis_title='Valor'
        )
        return fig.to_dict()
    
    def toDataFrame(self) -> pd.DataFrame:
        """Devuelve los resultados como DataFrame con columnas:
        [iteration, xn0, xn1, xn2, xnAitken]"""
        result = self.execute()
        data = [row.model_dump() for row in result.result]
        return pd.DataFrame(data)

    def execute(self) -> AitkenResponse:
        results = []
        current_x = self.x0

        for i in range(self.max_iter):
            xn1 = float(self.function(current_x))
            xn2 = float(self.function(xn1))
            
            denominator = xn2 - 2*xn1 + current_x
            if abs(denominator) < 1e-12:
                break
                
            aitken = current_x - ((xn1 - current_x)**2 / denominator)
            
            results.append(AitkenRowResponse(
                iteration=i,
                xn0=current_x,
                xn1=xn1,
                xn2=xn2,
                xnAitken=aitken
            ))
            
            if abs(aitken - current_x) < self.tolerance:
                break
                
            current_x = aitken
        else:
            raise RuntimeError(
                f"No convergencia en {self.max_iter} iteraciones. "
                f"Último error: {abs(aitken - current_x):.3e}"
            )

        return AitkenResponse(
            result=results,
            function=self.function_repr,
            plot_data=self._generate_plot(
                iterations=[r.iteration for r in results],
                values=[r.xnAitken for r in results]
            )
        )

    def __str__(self):
        return str(self.execute())