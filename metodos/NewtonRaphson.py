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

class NewtonRaphsonRequest(BaseModel):
    function: str
    x0: float
    tolerance: float = None
    max_iterations: int = None

    @field_validator('x0')
    def validate_x0(cls, v: float):
        """El punto inicial debe ser finito para garantizar convergencia"""
        if not np.isfinite(v):
            raise ValueError("x0 debe ser un número finito")
        return v
    
class NewtonRaphsonRowResponse(BaseModel):
    iteration: int
    xn: float
    f_xn: float
    df_xn: float
    xn1: float

class NewtonRaphsonResponse(BaseModel):
    result: List[NewtonRaphsonRowResponse]
    root: float
    function: str
    derivative: str
    plot_data: Dict[str, Any]

class NewtonRaphsonCalculator:
    def __init__(self, request: NewtonRaphsonRequest):
        self.x0 = request.x0
        self.tolerance = abs(request.tolerance) if request.tolerance else TOLERANCIA
        self.max_iter = max(1, request.max_iterations or MAX_ITER)
        self.function, self.function_repr, self.derivative, self.derivative_repr = self._setup_functions(request.function)

    def _setup_functions(self, func_str: str) -> Tuple[Callable, str, Callable, str]:
        x = sp.symbols('x')
        try:
            expr = sp.sympify(func_str)
            deriv_expr = sp.diff(expr, x)
            return (
                sp.lambdify(x, expr, modules=MODULES['numpy']),
                str(expr),
                sp.lambdify(x, deriv_expr, modules=['numpy']),
                str(deriv_expr)
            )
        except sp.SympifyError:
            raise ValueError(f"Función inválida: {func_str}")

    def _generate_plot(self, iterations: List[int], values: List[float]) -> Dict[str, Any]:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=iterations,
            y=values,
            mode='lines+markers',
            name='Convergencia',
            line=dict(color='royalblue', width=2)
        ))
        fig.update_layout(
            title='Método de Newton-Raphson',
            xaxis_title='Iteración',
            yaxis_title='Valor de x',
            hovermode='x unified'
        )
        return fig.to_dict()
    
    def toDataFrame(self) -> pd.DataFrame:
        """Devuelve los resultados como DataFrame con columnas:
        [iteration, xn, f_xn, df_xn, xn1]"""
        result = self.execute()
        data = [row.model_dump() for row in result.result]
        return pd.DataFrame(data).to_string(index=False)

    def execute(self) -> NewtonRaphsonResponse:
        results = []
        current_x = self.x0

        for i in range(self.max_iter):
            fx = self.function(current_x)
            dfx = self.derivative(current_x)
            xn1 = current_x - fx / dfx
            
            results.append(NewtonRaphsonRowResponse(
                iteration=i,
                xn=current_x,
                f_xn=fx,
                df_xn=dfx,
                xn1=xn1
            ))
            
            if abs(xn1 - current_x) < self.tolerance:
                break
                
            current_x = xn1
        else:
            raise RuntimeError(f"No convergencia en {self.max_iter} iteraciones. Último error: {abs(xn1 - current_x):.3e}")

        return NewtonRaphsonResponse(
            result=results,
            root=current_x,
            function=self.function_repr,
            derivative=self.derivative_repr,
            plot_data=self._generate_plot(
                iterations=[r.iteration for r in results],
                values=[r.xn for r in results]
            )
        )

    def __str__(self):
        return str(self.execute())