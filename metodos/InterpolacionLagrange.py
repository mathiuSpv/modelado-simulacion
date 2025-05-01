try:
    from . import pd, np, go, sp
except ImportError:
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    import sympy as sp

from pydantic import BaseModel
from typing import List, Dict, Any

class LagrangeRequest(BaseModel):
    x_puntos: List[float]
    y_puntos: List[float]

class LagrangeResponse(BaseModel):
    polinomio: str
    plot_data: Dict[str, Any]

class LagrangeCalculator:
    def __init__(self, request: LagrangeRequest):
        self.x_puntos = np.array(request.x_puntos)
        self.y_puntos = np.array(request.y_puntos)
        self._validar_puntos()

    def _validar_puntos(self):
        if len(self.x_puntos) != len(self.y_puntos):
            raise ValueError("x_puntos y y_puntos deben tener la misma longitud")

    def _interpolar(self) -> np.poly1d:
        n = len(self.x_puntos)
        polinomio = np.poly1d(0.0)
        for i in range(n):
            term = np.poly1d([1.0])
            for j in range(n):
                if i != j:
                    term *= np.poly1d([1, -self.x_puntos[j]]) / (self.x_puntos[i] - self.x_puntos[j])
            polinomio += self.y_puntos[i] * term
        return polinomio

    def _generate_plot(self, polinomio: np.poly1d) -> Dict[str, Any]:
        x_vals = np.linspace(min(self.x_puntos)-1, max(self.x_puntos)+1, 400)
        y_vals = polinomio(x_vals)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.x_puntos, y=self.y_puntos, mode='markers', name='Puntos dados'
        ))
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals, mode='lines', name='Polinomio de Lagrange'
        ))
        fig.update_layout(
            title='Interpolación de Lagrange',
            xaxis_title='x',
            yaxis_title='P(x)'
        )
        return fig.to_dict()

    def execute(self) -> LagrangeResponse:
        polinomio = self._interpolar()
        return LagrangeResponse(
            polinomio=str(polinomio),
            plot_data=self._generate_plot(polinomio)
        )

    def __str__(self):
        return str(self.execute())