try:
    from . import pd, np, go, sp, TOLERANCIA, MAX_ITER
except ImportError:
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    import sympy as sp
    TOLERANCIA = 1e-9
    MAX_ITER = 100

from pydantic import BaseModel
from typing import List, Dict, Any, Callable

class PuntoFijoRequest(BaseModel):
    function: str
    x0: float
    tolerance: float
    max_iterations: int

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
        self._validate_inputs()

    def _validate_inputs(self):
        """Valida que x0 sea finito y tolerance positivo."""
        if not np.isfinite(self.x0):
            raise ValueError(f"x0 debe ser finito. Se recibió: {self.x0}")
        if self.tolerance <= 0:
            raise ValueError(f"Tolerancia debe ser > 0. Se recibió: {self.tolerance}")

    def _setup_function(self, func_str: str) -> tuple[Callable, str]:
        x = sp.symbols('x')
        try:
            expr = sp.sympify(func_str)
            deriv = sp.diff(expr, x)
            deriv_val = deriv.subs(x, self.x0)
            if abs(float(deriv_val)) >= 1:
                raise ValueError(
                    f"|g'({self.x0})| = {abs(float(deriv_val)):.2f} ≥ 1\n"
                    f"Sugerencia: Modifique la función g(x)"
                )
            return (
                sp.lambdify(x, expr, modules=['numpy']),
                str(expr)
            )
        except sp.SympifyError:
            raise ValueError(f"Función inválida: {func_str}")

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