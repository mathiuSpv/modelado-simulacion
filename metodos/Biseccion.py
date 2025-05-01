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
from typing import List, Dict, Any

class BiseccionRequest(BaseModel):
    function: str
    a: float
    b: float
    tolerance: float = None
    max_iterations: int = None

    @field_validator('b')
    def validate_interval_and_sign(cls, v: float, values: Any):
        """Valida que a < b y que f(a)*f(b) < 0"""
        if 'a' not in values.data:
            return v
            
        if v <= values.data['a']:
            raise ValueError("b debe ser mayor que a")
        
        try:
            x = sp.symbols('x')
            expr = sp.sympify(values.data['function'])
            func = sp.lambdify(x, expr, modules=MODULES)
            a, b = values.data['a'], v
            fa = func(a)
            fb = func(b)
            
            if fa * fb >= 0:
                raise ValueError(
                    f"Funcion: {expr}\n"
                    f"No hay cambio de signo en [a, b]. f({a})={fa:.3f}, f({b})={fb:.3f}\n"
                    f"Requisito: f(a)*f(b) < 0"
                )
        except sp.SympifyError:
            raise ValueError("Función inválida: no se puede parsear para validación")
            
        return v

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
        self.function = self._parse_function(request.function)

    def _parse_function(self, func_str: str) -> callable:
        x = sp.symbols('x')
        expr = sp.sympify(func_str)
        return sp.lambdify(x, expr, modules=MODULES)

    def _generate_plot(self, raiz: float) -> Dict[str, Any]:
        x_vals = np.linspace(self.a - 1, self.b + 1, 400)
        y_vals = [self.function(xi) for xi in x_vals]
        
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
    
    def toDataFrame(self) -> pd.DataFrame:
        """Devuelve los resultados como DataFrame con columnas: 
        [iteracion, a, b, c, f_c]"""
        result = self.execute()
        data = [row.model_dump() for row in result.resultado]
        return pd.DataFrame(data).to_string(index=False)

    def execute(self) -> BiseccionResponse:
        resultados = []
        a, b = self.a, self.b
        
        for i in range(self.max_iter):
            c = (a + b) / 2
            f_c = self.function(c)
            
            resultados.append(BiseccionRowResponse(
                iteracion=i,
                a=a,
                b=b,
                c=c,
                f_c=f_c
            ))
            
            if abs(f_c) < self.tol or (b - a)/2 < self.tol:
                break
                
            if self.function(a) * f_c < 0:
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
    
if __name__ == "__main__":
    import __config__
    from metodos.test.pdf1Biseccion import testCases
    
    testCases()