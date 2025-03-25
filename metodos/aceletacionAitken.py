from pandas import DataFrame
from errors.params import paramsHandlerAitken

@paramsHandlerAitken
def aceleracionAitken(df: DataFrame) -> DataFrame:
    """
    Herramienta utilizada para acelerar el proceso de iteracion del **Metodo del Punto Fijo**.
    Se utiliza una funcion matematica para hacer el calculo, para ello es necesario 3 valores (Ver **Parameters**)
    Mas informacion en:
    https://es.wikipedia.org/wiki/Proceso_%CE%94%C2%B2_de_Aitken
    
    :param df: Un DataFrame que debe contener exactamente tres columnas:\n
              - 'x_n0': Termino de Sucesion Original X(n).
              - 'x_n1': Resultado de Iteracion x_n1 = g(x_n0).
              - 'x_n2': Resultado de Iteracion x_n2 = g(x_n1).
              Cada columna debe contener un solo valor (la primera fila del DataFrame).
    :type df: DataFrame

    :return: El mismo DataFrame de entrada con una columna adicional llamada
             'x_nA', que contiene el resultado del **termino de la sucesion acelerada**: x*n.
    :rtype: DataFrame

    :raises ValueError: Si el DataFrame no tiene exactamente 3 columnas || si no contiene
                       las columnas 'x_n0', 'x_n1' y 'x_n2'.
    """
    # Extraer los valores de las columnas
    x_n0 = df['x_n0'].iloc[0]
    x_n1 = df['x_n1'].iloc[0]
    x_n2 = df['x_n2'].iloc[0]
    
    # Operacion Aitken 
    resultado_aitken = x_n0 - ((x_n1 - x_n0)**2 / (x_n2 - 2 * x_n1 + x_n0))
    df['x_nA'] = [resultado_aitken]
    return df