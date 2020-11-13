# Correspondente a segunda tarefa do EP de Computação III

import math

import numpy as np

A = [[3 / 10, 3 / 5, 0], [1 / 2, 0, 1], [4 / 10, 4 / 5, 0]]
An = np.atleast_2d(A).shape[0]
Am = np.atleast_2d(A).shape[1]

# An = Wn
# Am = Hm
# Wm = An - Arbitrario

Arb = 2

W = np.random.randint(100, size=(An, Arb))
H = np.empty((Arb, Am), dtype=float)
