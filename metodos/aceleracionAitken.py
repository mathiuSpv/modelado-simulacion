from __config__ import *
from error.condicion import conditionHandlerPuntoFijo
from math import sqrt, pi

@conditionHandlerPuntoFijo
def aceleracionAitken(g, xn0: float, tol=1e-9, max_iter=100) -> pd.DataFrame:
    """
    Herramienta utilizada para acelerar el proceso de iteracion del **Metodo del Punto Fijo**.
    Se utiliza una funcion matematica para hacer el calculo, para ello es necesario 3 valores (Ver **Parameters**)
    Mas informacion en:
    https://es.wikipedia.org/wiki/Proceso_%CE%94%C2%B2_de_Aitken
    
    :param g: Función de iteración del método del punto fijo (x_{n+1} = g(x_n)).
    :type g: function

    :param xn0: Valor inicial para comenzar las iteraciones.
    :type xn0: float

    :param tol: Tolerancia para el criterio de parada (diferencia entre iteraciones).
    :type tol: float, opcional

    :param max_iter: Número máximo de iteraciones permitidas.
    :type max_iter: int, opcional

    :return: DataFrame con las siguientes columnas:
             - 'xn0': Términos originales de la sucesión (x_n)
             - 'xn1': Resultados de la primera iteración (x_{n+1} = g(x_n))
             - 'xn2': Resultados de la segunda iteración (x_{n+2} = g(x_{n+1}))
             - 'xnAitken': Resultados acelerados por el método de Aitken
    :rtype: pandas.DataFrame
    
    :raises ValueError: Si |g'(x0)| ≥ 1 (falla la condición de convergencia)\n
                       Si la función g(x) no es numéricamente estable\n
                       Si x0 no es un número finito

    :..note: La iteración se detiene cuando se alcanza la tolerancia o el máximo de iteraciones.
    """
    array_xn0 = []; array_xn1 = []; array_xn2 = []; array_xna = []
    for _ in range(max_iter):
        xn1 = g(xn0)
        xn2 = g(xn1)
        denominator = xn2 - 2 * xn1 + xn0
        if abs(denominator) < 1e-12:
            break
        aitken = xn0 - ((xn1 - xn0)**2 / denominator)
        
        array_xn0.append(xn0)
        array_xn1.append(xn1)
        array_xn2.append(xn2)
        array_xna.append(aitken)
        if abs(aitken - xn0) < tol:
            break
        xn0 = aitken
        
    return pd.DataFrame({
        'xn0': array_xn0,
        'xn1': array_xn1,
        'xn2': array_xn2,
        'xnAitken': array_xna
    })
    
if __name__ == "__main__":
    """Función a resolver: f(x) = (π/2)x² - x - 2"""    
    g_valida = lambda x: sqrt((2 * x + 4) / pi)
    
    print("=== Prueba con Ej1 ===")
    df1 = aceleracionAitken(g_valida, 1.4)
    print(df1.to_string(index=False))