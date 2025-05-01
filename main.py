import metodos.AceleradorAitken as mt
import metodos.PuntoFijo as pf
import metodos.NewtonRaphson as nr
import metodos.Biseccion as bs
import metodos.BusquedaBinaria as bi
import metodos.InterpolacionLagrange as li
import metodos.NewtonCotes as nc


binaria  = bi.BusquedaBinariaRequest(function="exp(x) - 3*x**2", a= 0, b= 1)
b_ = bi.BusquedaBinariaCalculator(binaria)
# print(b_.execute().root)

aitken = mt.AitkenRequest(function='sqrt(exp(x)/3)',x0=0.5, tolerance=1e-6)

print(mt.AitkenCalculator(aitken).toDataFrame())

newtonrapson = nr.NewtonRaphsonRequest(function='exp(x) - 3*x**2', x0=.5, tolerance= 1e-8)
# print(nr.NewtonRaphsonCalculator(newtonrapson).toDataFrame())


""" Punto 2 (No lo logramos)
a = li.LagrangeRequest(x_puntos=[0,1,2], y_puntos=[0, 0.6931471806, 1.098612289])
print(li.LagrangeCalculator(a).toDataFrame())
"""

""" Punto 3

"""
from metodos.NewtonCotes import NewtonCotesRequest, NewtonCotesCalculator, IntegrationMethod
import numpy as np
import sympy as sp

# Definimos la función
def f(x):
    return np.sqrt(2 * np.exp(x**2))

# Parte a) - Regla del Trapecio
def parte_a():
    xi = 0.5
    n_values = [4, 10]
    
    for n in n_values:
        # Creamos la solicitud
        request = NewtonCotesRequest(
            function="sqrt(2*exp(x**2))",
            lower_bound=0,
            upper_bound=xi,
            num_intervals=n,
            method=IntegrationMethod.COMPOUND_TRAPEZOID
        )
        
        # Calculamos
        calculator = NewtonCotesCalculator(request)
        result = calculator.execute()
        
        # Cálculo del valor real usando sympy
        x = sp.symbols('x')
        real_integral = sp.integrate(sp.sqrt(2*sp.exp(x**2)), (x, 0, xi))
        real_value = float(real_integral)
        
        # Error
        error = abs(real_value - result.result)
        
        print(f"\nPara n={n}:")
        print(f"  Aproximación: {result.result}")
        print(f"  Valor real: {real_value}")
        print(f"  Error: {error}")

parte_a()