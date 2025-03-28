from __config__ import *
from error.condicion import conditionHandlerPuntoFijo
from math import sqrt, pi

@conditionHandlerPuntoFijo
def metodoPuntoFijo(g, x0: float, tol=1e-9, max_iter=100) -> pd.DataFrame:
    """
    Herramienta para resolver ecuaciones no lineales mediante el **Metodo del Punto Fijo** iterativo.
    Aproxima la solución de ecuaciones de la forma x = g(x) mediante iteraciones sucesivas.
    
    :param g: Función de iteración que define la relación x = g(x). Debe cumplir:\n
              - Ser continua y diferenciable en un intervalo [a, b]\n
              - Cumplir |g'(x)| < 1 para todo x en [a, b]\n
    :type g: function
    
    :param x0: Valor inicial para comenzar las iteraciones. Debe estar cercano a la raíz buscada.
    :type x0: float
    
    :param tol: (Opcional) Tolerancia para el criterio de parada (|x_{n+1} - x_n| < tol). 
                Valor por defecto: 1e-6.
    :type tol: float
    
    :param max_iter: (Opcional) Número máximo de iteraciones permitidas.
                     Valor por defecto: 100.
    :type max_iter: int
    
    :return: DataFrame con los resultados de cada iteración, contiene:\n
             - 'Iteración': Número de iteración (desde 0 hasta n)\n
             - 'xn': Valor actual de x en la iteración\n
             - 'g(xn)': Valor de la siguiente iteración\n
    :rtype: pandas.DataFrame
    
    :raises ValueError: Si |g'(x0)| ≥ 1 (falla la condición de convergencia)\n
                       Si la función g(x) no es numéricamente estable\n
                       Si x0 no es un número finito
    
    :raises RuntimeError: Si no se alcanza la convergencia después de max_iter iteraciones
    """
    iter = []  # Número de iteración
    val_xn = []   # Guarda el valor actual
    val_gxn = []  # Guarda el valor g(x_n)
    for i in range(max_iter):
        xn1 = g(x0)  # Calculamos x_{n+1}
        
        # Registramos los datos
        iter.append(i)
        val_xn.append(x0)
        val_gxn.append(xn1)
        
        if abs(xn1 - x0) < tol:
            break
        x0 = xn1
    return pd.DataFrame({
        'i': iter,
        'xn': val_xn,
        'g(xn)': val_gxn
    })


if __name__ == "__main__":
    """Función a resolver: f(x) = (π/2)x² - x - 2"""    
    g_valida = lambda x: sqrt((2 * x + 4) / pi)
    g_no_valida = lambda x: (pi / 2) * x**2 - 2
    
    print("=== Prueba con g_valida ===")
    try:
        df1 = metodoPuntoFijo(g_valida, 1.4)
        print(df1.to_string(index=False))
    except Exception as e:
        print(f"Error: {e}")

    print("\n=== Prueba con g_no_valida ===")
    try:
        df2 = metodoPuntoFijo(g_no_valida, 1.4)
        print(df2.to_string(index=False))
    except ValueError as e:
        print(f"✅ Error capturado:\n{e}")