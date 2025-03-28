import numpy as np
import matplotlib.pyplot as plt

def polinomio_lagrange(x, x_puntos, y_puntos):
    n = len(x_puntos)
    L = 0
    for i in range(n):
        # Calcular el polinomio base de Lagrange
        li = 1
        for j in range(n):
            if i != j:
                li *= (x - x_puntos[j]) / (x_puntos[i] - x_puntos[j])
        L += y_puntos[i] * li
    return L

def reconstruccion_lagrange(x_puntos, y_puntos):
    n = len(x_puntos)
    coeficientes = np.zeros(n)
    for i in range(n):
        # Calcular el polinomio base de Lagrange
        li = np.poly1d([1])
        for j in range(n):
            if i != j:
                li *= np.poly1d([1, -x_puntos[j]]) / (x_puntos[i] - x_puntos[j])
        coeficientes += y_puntos[i] * li.coefficients
    return coeficientes

# Definir los puntos dados
x_puntos = np.array([1,2,3])
y_puntos = np.array([1,4,9])

# Crear un conjunto de valores x para graficar
x = np.linspace(min(x_puntos) - 1, max(x_puntos) + 1, 400)
y = [polinomio_lagrange(xi, x_puntos, y_puntos) for xi in x]

# Graficar los puntos dados
plt.scatter(x_puntos, y_puntos, color='red', label='Puntos dados')

# Graficar el polinomio de Lagrange
plt.plot(x, y, label='Polinomio de Lagrange')

# Añadir leyenda y mostrar la gráfica
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Interpolación de Lagrange')
plt.grid(True)
plt.show()

# Reconstruir el polinomio de Lagrange
coeficientes = reconstruccion_lagrange(x_puntos, y_puntos)
polinomio = np.poly1d(coeficientes)

# Imprimir el polinomio reconstruido
print(f"El polinomio de Lagrange es:\n{polinomio}")

