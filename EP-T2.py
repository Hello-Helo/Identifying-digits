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

###############################################################################


def Normalize(W):
    Wn = np.atleast_2d(W).shape[0]
    Wm = np.atleast_2d(W).shape[1]
    for j in range(0, Wn):
        s = math.sqrt((np.sum(W[0:Wn, j]**2))
        for n in range(0, An):
            W[n, m] = W[n, m] / Sj
        W[0: Wn, j] = W[0: Wn, j]/s
    return W

###############################################################################


def Solution(W, b):
    # Tmanho das matrizes
    Wm = np.atleast_2d(W).shape[1]
    bm = np.atleast_2d(b).shape[1]
    # Solução
    x = np.empty((Wm, bm), dtype=float)
    for w in range(0, bm):
        for k in range(Wm - 1, -1, -1):
            som = 0
            for j in range(k + 1, Wm):
                som = som + W[k, j] * x[j, w]
            x[k, w] = (b[k, w] - som) / W[k, k]
    return x


###############################################################################


def Make_positive(W):
    Wn = np.atleast_2d(W).shape[0]
    Wm = np.atleast_2d(b).shape[1]
        for i in range(0, Wn):
            for j in range(0, Wm):
                W[i, j] = max(0, W[1, j])
    return W


###############################################################################


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

while E > 0.000001 and iterations < 100:
    W = Normalize(W)

    Transf = EP-T1.Transformation(W, A)
    W = Transf[0]
    A = Transf[1]
    H = Solution(W, A)

    A = Aprime

    H = Make_positive(H)

    At = np.transpose(A)
    Ht = np.transpose(H)

    Transf = EP-T1.Transformation(Ht, At)
    Ht = Transf[0]
    At = Transf[1]
    Wt = Solution(Ht, At)

    A = Aprime

    W = np.transpose(Wt)
    W = Make_positive(W)

    E = (A - W*H)**2
    iterations += 1
