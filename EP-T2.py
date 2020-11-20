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


def Transformation(W, b):
    Wn = np.atleast_2d(W).shape[0]
    Wm = np.atleast_2d(W).shape[1]
    for k in range(0, Wm):
        for j in range(Wn - 1, k, -1):
            i = j - 1
            if W[j, k] != 0:
                const = Constants(W, j, k, i)
                s = const[0]
                c = const[1]
                W = Rotgivens(W, Wn, Wm, i, j, c, s)
                b = Rotgivens(b, Wn, Wm, i, j, c, s)
    return W, b


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


def Normalize(W):
    Wn = np.atleast_2d(W).shape[0]
    Wm = np.atleast_2d(W).shape[1]
    for j in range(0, Wm):
        s = math.sqrt((np.sum(W[0:Wn, j] ** 2)))
        # for i in range(0, An):
        #     W[i, j] = W[i, j] / s
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
    WH = np.matmul(W, H)
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
print(A, end="\n\n")

Aprime = np.copy(A)

E = 10
Edif = 10
iterations = 0

while Edif > 0.000001 and iterations < 100:
    print("ITERATION", iterations)
    # print(W)
    W = Normalize(W)

    Transf = Transformation(W, A)
    W = Transf[0]
    A = Transf[1]
    # print("A matriz W:")
    # print(W, end="\n")
    # print("A matriz A:")
    # print(A, end="\n")
    H = Solution(W, A)

    A = np.copy(Aprime)

    # print("A matriz A:")
    # print(A, end="\n")

    # print("A matriz H:")
    # print(H, end="\n")

    H = Make_positive(H)
    # print("A matriz H:")
    # print(H, end="\n")
    At = np.transpose(A)
    Ht = np.transpose(H)

    Transf = Transformation(Ht, At)
    Ht = Transf[0]
    At = Transf[1]
    # print("A matriz Ht:")
    # print(Ht, end="\n")
    # print("A matriz At:")
    # print(A, end="\n")
    Wt = Solution(Ht, At)

    A = np.copy(Aprime)

    W = np.transpose(Wt)
    W = Make_positive(W)
    # print("A matriz W:")
    # print(W, end="\n")

    # print("A matriz WH:")
    # print(np.matmul(W, H), end="\n")
    # print("A matriz A:")
    # print(A, end="\n")
    Eprev = E
    E = Erro(A, W, H)
    Edif = abs(E - Eprev)
    print("ERROR = ", E, end="\n\n")
    iterations += 1
