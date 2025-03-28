from functools import wraps

def conditionHandlerPuntoFijo(func):
    """Decorador que verifica |g'(x0)| < 1"""
    @wraps(func)
    def wrapper(g, x0, *args, **kwargs):
        h = 1e-6  # Desborde para el cálculo de la derivada.
        # Calculamos con Diferencial Finitas: g'(x0) ≈ [g(x0 + h) - g(x0)] / h
        g_ = (g(x0 + h) - g(x0)) / h
        
        if abs(g_) >= 1:
            raise ValueError(
                f"Condición fallida: |g'({x0})| = {abs(g_):.2f} < 1\n"
                f"Valor de g'(x0) = {g_}\n"
                "Sugerencia: Modifica g(x)."
            )
        return func(g, x0, *args, **kwargs)
    return wrapper