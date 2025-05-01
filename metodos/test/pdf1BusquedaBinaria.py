from metodos.BusquedaBinaria import BusquedaBinariaRequest, BusquedaBinariaCalculator
from typing import List, Dict

requests: List[Dict] = []

# ===== EJERCICIO 1 =====
requests.append({
    'ejercicio': '1a',
    'request': BusquedaBinariaRequest(
        function="exp(x) - 2 - x",
        a=0.0,
        b=1.0,
        tolerance=1e-6,
        max_iterations=100
    ),
    'descripcion': "f(x)=e^x-2-x en [0,1] (Hallar intervalo)"
})

requests.append({
    'ejercicio': '1b',
    'request': BusquedaBinariaRequest(
        function="cos(x) + x",
        a=-1.0,
        b=0.0,
        tolerance=1e-6,
        max_iterations=100
    ),
    'descripcion': "f(x)=cos(x)+x en [-1,0]"
})

requests.append({
    'ejercicio': '1c',
    'request': BusquedaBinariaRequest(
        function="log(x) - 5 + x",
        a=3.0,
        b=4.0,
        tolerance=1e-6,
        max_iterations=100
    ),
    'descripcion': "f(x)=ln(x)-5+x en [3,4]"
})

requests.append({
    'ejercicio': '1d',
    'request': BusquedaBinariaRequest(
        function="x**2 - 10*x + 23",
        a=3.0,
        b=7.0,
        tolerance=1e-6,
        max_iterations=100
    ),
    'descripcion': "f(x)=x²-10x+23 en [3,7]"
})

def testCases():
    """Ejecuta y muestra los resultados básicos de cada ejercicio"""
    for caso in requests:
        try:
            print(f"\n=== EJERCICIO {caso['ejercicio']} ===")
            print(f"Descripción: {caso['descripcion']}")
            
            calculator = BusquedaBinariaCalculator(caso['request'])
            resultado = calculator.execute()
            df = calculator.toDataFrame()
            
            print(f"\nRaíz encontrada: {resultado.root:.8f}")
            print(f"Iteraciones: {len(df)}")
            print("\nTabla de iteraciones:")
            print(df.to_string(index=False))
            
        except Exception as e:
            print(f"\nError en ejercicio {caso['ejercicio']}: {str(e)}")