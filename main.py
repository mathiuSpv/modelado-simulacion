import metodos.AceleradorAitken as mt
import metodos.PuntoFijo as pf
import metodos.NewtonRaphson as nr
import metodos.Biseccion as bs
import metodos.BusquedaBinaria as bi

biseccion = bs.BiseccionRequest(function='exp(x) - 3*x**2', a= 0, b= 1)
binaria  = bi.BusquedaBinariaRequest(function="exp(x) - 3*x**2", a= 0, b= 1)
b_ = bi.BusquedaBinariaCalculator(binaria)
a_ = bs.BiseccionCalculator(biseccion)
print(a_.toDataFrame())
print(b_.toDataFrame())

puntofijo = pf.PuntoFijoRequest(function='sqrt(exp(x)/3)',x0=-0.28172)
aitken = mt.AitkenRequest(function='sqrt(exp(x)/3)',x0=-0.28172)

a= mt.AitkenCalculator(aitken)
print(a.toDataFrame())
# b= pf.PuntoFijoCalculator(puntofijo)
# print(b.toDataFrame())

newtonrapson = nr.NewtonRaphsonRequest(function='sqrt(exp(x)/3)', x0=0.5)