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
from typing import List, Dict, Any

class BiseccionRequest(BaseModel):
    funcion: str
    a: float
    b: float
    tolerance: float
    max_iterations: int

class BiseccionRowResponse(BaseModel):
    iteracion: int
    a: float
    b: float
    c: float
    f_c: float

class BiseccionResponse(BaseModel):
    resultado: List[BiseccionRowResponse]
    raiz: float
    plot_data: Dict[str, Any]

class BiseccionCalculator:
    def __init__(self, request: BiseccionRequest):
        self.a = request.a
        self.b = request.b
        self.tol = request.tolerance or TOLERANCIA
        self.max_iter = max(1, request.max_iterations or MAX_ITER)
        self.funcion = self._parse_function(request.funcion)
        self._validar_intervalo()

    def _parse_function(self, func_str: str) -> callable:
        x = sp.symbols('x')
        expr = sp.sympify(func_str)
        return sp.lambdify(x, expr, modules=['numpy', 'math'])

    def _validar_intervalo(self):
        if self.funcion(self.a) * self.funcion(self.b) >= 0:
            raise ValueError("f(a) y f(b) deben tener signos opuestos")

    def _generate_plot(self, raiz: float) -> Dict[str, Any]:
        x_vals = np.linspace(self.a - 1, self.b + 1, 400)
        y_vals = [self.funcion(xi) for xi in x_vals]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals, mode='lines', name='Función'
        ))
        fig.add_trace(go.Scatter(
            x=[raiz], y=[0], mode='markers', name='Raíz'
        ))
        fig.update_layout(
            title='Método de Bisección',
            xaxis_title='x',
            yaxis_title='f(x)'
        )
        return fig.to_dict()

    def execute(self) -> BiseccionResponse:
        resultados = []
        a, b = self.a, self.b
        
        for i in range(self.max_iter):
            c = (a + b) / 2
            f_c = self.funcion(c)
            
            resultados.append(BiseccionRowResponse(
                iteracion=i,
                a=a,
                b=b,
                c=c,
                f_c=f_c
            ))
            
            if abs(f_c) < self.tol or (b - a)/2 < self.tol:
                break
                
            if self.funcion(a) * f_c < 0:
                b = c
            else:
                a = c
        
        return BiseccionResponse(
            resultado=resultados,
            raiz=c,
            plot_data=self._generate_plot(c)
        )

    def __str__(self):
        return str(self.execute())