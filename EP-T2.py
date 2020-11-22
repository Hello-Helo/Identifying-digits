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


def Rotgivens(W, n, m, i, j, c, s):
    W[i, 0:m], W[j, 0:m] = c * W[i, 0:m] - s * W[j, 0:m], s * W[i, 0:m] + c * W[j, 0:m]
    return W


###############################################################################


def Is_tsup(W, Wn, Wm):
    is_valid = True
    for collum in range(0, Wm):
        for line in range(collum + 1, Wn):
            if W[line][collum] != 0:
                is_valid = False
                return is_valid
    return is_valid


###############################################################################


def Constants(W, j, k, i):
    if abs(W[i, k]) > abs(W[j, k]):
        t = -W[j, k] / W[i, k]
        c = 1 / math.sqrt(1 + t ** 2)
        s = c * t
    else:
        t = -W[i, k] / W[j, k]
        s = 1 / math.sqrt(1 + t ** 2)
        c = s * t
    return s, c


###############################################################################


def Solution(W, b):

    Wn = W.shape[0]
    Wm = W.shape[1]
    bn = b.shape[0]
    bm = b.shape[1]
    for k in range(0, Wm):
        for j in range(Wn - 1, k, -1):
            i = j - 1
            if W[j, k] != 0:
                const = Constants(W, j, k, i)
                s = const[0]
                c = const[1]
                W = Rotgivens(W, Wn, Wm, i, j, c, s)
                b = Rotgivens(b, bn, bm, i, j, c, s)

    # Tamanho das matrizes
    Wm = W.shape[1]
    bm = b.shape[1]
    # Solução
    x = np.zeros((Wm, bm), dtype=float)
    for w in range(0, bm):
        for k in range(Wm - 1, -1, -1):
            som = 0
            for j in range(k + 1, Wm):
                som = som + W[k, j] * x[j, w]
            x[k, w] = (b[k, w] - som) / W[k, k]
    return x


###############################################################################


def Normalize(W):
    Wn = np.atleast_2d(W).shape[0]
    Wm = np.atleast_2d(W).shape[1]
    for j in range(0, Wm):
        s = math.sqrt((np.sum(W[0:Wn, j] ** 2)))
        W[0:Wn, j] = W[0:Wn, j] / s
    return W


###############################################################################


def Make_positive(W):
    Wn = np.atleast_2d(W).shape[0]
    Wm = np.atleast_2d(W).shape[1]
    for i in range(0, Wn):
        for j in range(0, Wm):
            W[i, j] = max(0, W[i, j])
    return W


###############################################################################


def Erro(A, W, H):
    erro = 0
    WH = np.dot(W, H)
    An = np.atleast_2d(A).shape[0]
    Am = np.atleast_2d(A).shape[1]
    for i in range(0, An):
        for j in range(0, Am):
            erro = erro + (A[i, j] - WH[i, j]) ** 2
    return erro

###############################################################################


A = np.array([[3 / 10, 3 / 5, 0], [1 / 2, 0, 1], [4 / 10, 4 / 5, 0]])
An = np.atleast_2d(A).shape[0]
Am = np.atleast_2d(A).shape[1]

# An = Wn
# Am = Hm
# Wm = An - Arbitrario

Arb = 2

W = np.random.random(size=(An, Arb))
H = np.empty((Arb, Am), dtype=float)

print("A matriz A original:")
print(A, end="\n")

Aprime = np.copy(A)

E = 10
iterations = 0

while E > 0.000001 and iterations < 20:
    print("ITERATIONS ", end ="\n\n")
    print(W)
    W = Normalize(W)
 
    H = Solution(W, A)
    print(np.dot(W,H))
    
    A = np.copy(Aprime)

    H = Make_positive(H)

    At = np.transpose(np.copy(A))
    Ht = np.transpose(np.copy(H))

    Wt = Solution(Ht, At)

    A = np.copy(Aprime)

    W = np.transpose(Wt)
    W = Make_positive(W)

    E = E - Erro(A, W, H)
    
    iterations += 1
    print(iterations)
