try:
    from ..metodos import pd, np, go, sp, TOLERANCIA, MAX_ITER
except ImportError:
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    import sympy as sp
    TOLERANCIA = 1e-9
    MAX_ITER = 100

from enum import Enum
from pydantic import BaseModel, field_validator
from typing import List, Dict, Any, Tuple, Callable, Optional

class ODEMethod(Enum):
    EULER = "Euler"
    HEUN = "Heun"
    RUNGE_KUTTA_4 = "Runge_Kutta_4"

class ODERequest(BaseModel):
    equation: str
    y0: float
    t0: float
    tf: float
    h: float
    exact_solution: Optional[str] = None

    @field_validator('tf')
    def validate_time_interval(cls, v, values):
        """El intervalo temporal debe ser válido (t0 < tf)"""
        if 't0' in values.data and v <= values.data['t0']:
            raise ValueError("tf debe ser mayor que t0")
        return v

    @field_validator('h')
    def validate_step_size(cls, v):
        """El paso h debe ser positivo para la integración numérica"""
        if v <= 0:
            raise ValueError("h debe ser positivo")
        return abs(v)

class ODEResponsePoint(BaseModel):
    t: float
    exact: Optional[float] = None
    euler: Optional[float] = None
    heun: Optional[float] = None
    rk4: Optional[float] = None

class ODEResponse(BaseModel):
    result: List[ODEResponsePoint]
    function: str
    plot_data: Dict[str, Any]
    status: str
    message: Optional[str] = None

class ODECalculator:
    def __init__(self, request: ODERequest):
        self.y0 = request.y0
        self.t0 = request.t0
        self.tf = request.tf
        self.h = abs(request.h) if request.h else TOLERANCIA
        self.function, self.function_repr = self._setup_function(request.equation)
        self.exact_solution = request.exact_solution

    def _setup_function(self, func_str: str) -> Tuple[Callable, str]:
        t, y = sp.symbols('t y')
        try:
            expr = sp.sympify(func_str)
            return (
                sp.lambdify((t, y), expr, modules=['numpy']),
                str(expr)
            )
        except sp.SympifyError:
            raise ValueError(f"Ecuación inválida: {func_str}")

    def _euler_method(self) -> Tuple[np.ndarray, np.ndarray]:
        t_vals = np.arange(self.t0, self.tf + self.h, self.h)
        y_vals = np.zeros_like(t_vals)
        y_vals[0] = self.y0
        for i in range(len(t_vals)-1):
            y_vals[i+1] = y_vals[i] + self.h * self.function(t_vals[i], y_vals[i])
        return t_vals, y_vals

    def _heun_method(self) -> Tuple[np.ndarray, np.ndarray]:
        t_vals = np.arange(self.t0, self.tf + self.h, self.h)
        y_vals = np.zeros_like(t_vals)
        y_vals[0] = self.y0
        for i in range(len(t_vals)-1):
            k1 = self.function(t_vals[i], y_vals[i])
            k2 = self.function(t_vals[i]+self.h, y_vals[i]+self.h*k1)
            y_vals[i+1] = y_vals[i] + self.h * (k1 + k2) / 2
        return t_vals, y_vals

    def _rk4_method(self) -> Tuple[np.ndarray, np.ndarray]:
        t_vals = np.arange(self.t0, self.tf + self.h, self.h)
        y_vals = np.zeros_like(t_vals)
        y_vals[0] = self.y0
        for i in range(len(t_vals)-1):
            h = self.h
            t, y = t_vals[i], y_vals[i]
            k1 = self.function(t, y)
            k2 = self.function(t + h/2, y + h*k1/2)
            k3 = self.function(t + h/2, y + h*k2/2)
            k4 = self.function(t + h, y + h*k3)
            y_vals[i+1] = y + h * (k1 + 2*k2 + 2*k3 + k4) / 6
        return t_vals, y_vals

    def _generate_plot(self, t_vals: np.ndarray, methods: Dict[str, np.ndarray]) -> Dict[str, Any]:
        fig = go.Figure()
        for name, y_vals in methods.items():
            fig.add_trace(go.Scatter(
                x=t_vals, y=y_vals,
                mode='lines+markers',
                name=name
            ))
        fig.update_layout(
            title='Solución de EDO',
            xaxis_title='t',
            yaxis_title='y(t)'
        )
        return fig.to_dict()
    
    def toDataFrame(self) -> pd.DataFrame:
        """Devuelve los resultados numéricos como DataFrame con columnas:
        [t, euler, heun, rk4, exact]"""
        result = self.execute()
        data = [row.model_dump() for row in result.result]
        return pd.DataFrame(data)

    def execute(self) -> ODEResponse:
        methods = {
            'euler': self._euler_method(),
            'heun': self._heun_method(),
            'rk4': self._rk4_method()
        }
        
        t_vals = methods['rk4'][0]
        results = []
        
        for i, t in enumerate(t_vals):
            point = ODEResponsePoint(t=t)
            for name, (_, y_vals) in methods.items():
                if i < len(y_vals):
                    setattr(point, name, float(y_vals[i]))
            results.append(point)

        return ODEResponse(
            result=results,
            function=self.function_repr,
            plot_data=self._generate_plot(t_vals, {k: v[1] for k, v in methods.items()}),
            status="success"
        )

    def __str__(self):
        return str(self.execute())