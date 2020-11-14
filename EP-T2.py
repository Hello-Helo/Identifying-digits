#      _   _ _
#     | | | | |        Heloisa Lazari & Lucca Alipio
#     | |_| | |        Ciências Moleculares - USP
#     |  _  | |___     heloisalbento@usp.br | luccapagnan@usp.br
#     |_| |_|_____|
#
#     Segunda tarefa do EP de Computação III
#
#
#
#
###############################################################################


import math

import numpy as np

A = [[3 / 10, 3 / 5, 0], [1 / 2, 0, 1], [4 / 10, 4 / 5, 0]]
An = np.atleast_2d(A).shape[0]
Am = np.atleast_2d(A).shape[1]

# An = Wn
# Am = Hm
# Wm = An - Arbitrario

Arb = 2

W = np.random.random(size=(An, Arb))
H = np.empty((Arb, Am), dtype=float)

Aprime = np.copy(A)

E = 10
iterations = 0

while E > 0.00001 and iterations < 100:
	for m in range(0, Arb):
		Sj = math.sqrt((np.sum(W[0:An, m]**2))
		for n in range(0, An):
			W[n, m] = W[n, m] / Sj

	c = T1ep.Transformation(W, A)
	W = c[0]
	A = c[1]
	H = T1ep.Solution(W, A)
	A = Aprime

	#Aqui temos que transformar H em Hi,j = max(Hi,j , 0)

    Ht = np.transpose(H)
    At = np.transpose(A)

    ct = T1ep.Transformation(Ht, At)
