 # Definir la función
import math
def f(x):
    return math.sin(x)
 # Punto y paso
x = 0.2
h = 0.1
 
# Diferencias finitas centradas para la primera derivada
def primera_derivada(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)
 
# Diferencias finitas centradas para la segunda derivada
def segunda_derivada(f, x, h):
    return (f(x + h) - 2*f(x) + f(x - h)) / (h**2)
 
# Calcular las derivadas
primera = primera_derivada(f, x, h)
segunda = segunda_derivada(f, x, h)
 
# Imprimir los resultados
print(f"Primera derivada en x = {x}: {primera}")
print(f"Segunda derivada en x = {x}: {segunda}")


