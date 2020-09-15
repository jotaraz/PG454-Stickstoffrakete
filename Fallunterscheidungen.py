import numpy as np
import matplotlib.pyplot as plt

def f(x): #Durch Fallunterschiedungen definierte Funktion
    if (x < 0):
        return x**2
    elif (x >= 0):
        return 0.1*x

def heaviside1(x, a): #ist 1, wenn x > a und 0, wenn x < a
    return 0.5*np.sign(x-a)+0.5


def heaviside2(x, a): #ist 1, wenn x < a und 0, wenn x > a
    return 0.5*np.sign(a-x)+0.5

def g(x): #Hat das gleiche Ergebnis, wie f(x), aber g(np.array([...]) ist berechenbar
    return heaviside2(x, 0)*x**2 + heaviside1(x, 0)*0.1*x

X = np.linspace(-1, 1, 1000)

#plt.plot(X, f(X)) #Diese Zeile erzeugt einen Error
plt.plot(X, g(X))
plt.show()
    
