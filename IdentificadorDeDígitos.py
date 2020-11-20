#      _   _ _
#     | | | | |        Heloisa Lazari & Lucca Alipio
#     | |_| | |        Ciências Moleculares - USP
#     |  _  | |___     heloisalbento@usp.br | luccapagnan@usp.br
#     |_| |_|_____|
#
#     Terceira tarefa do EP de Computação III
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


def Train(A, p):
    An = np.atleast_2d(A).shape[0]
    Am = np.atleast_2d(A).shape[1]
    W = np.random.random(size=(An, p))
    H = np.empty((p, Am), dtype=float)

    Aprime = np.copy(A)

    E = 10
    Edif = 10
    iterations = 0

    while Edif > 0.000001 and iterations < 100:
        print("ITERATION", iterations)
        W = Normalize(W)

        Transf = Transformation(W, A)
        W = Transf[0]
        A = Transf[1]
        H = Solution(W, A)

        A = np.copy(Aprime)

        H = Make_positive(H)
        At = np.transpose(A)
        Ht = np.transpose(H)

        Transf = Transformation(Ht, At)
        Ht = Transf[0]
        At = Transf[1]
        Wt = Solution(Ht, At)

        A = np.copy(Aprime)

        W = np.transpose(Wt)
        W = Make_positive(W)
        Eprev = E
        E = Erro(A, W, H)
        Edif = abs(E - Eprev)
        print("ERROR = ", E, end="\n\n")
        iterations += 1
    return W


###############################################################################

ndig = 50

A0 = np.loadtxt("train_dig0.txt", usecols=range(0, ndig))
A1 = np.loadtxt("train_dig1.txt", usecols=range(0, ndig))
A2 = np.loadtxt("train_dig2.txt", usecols=range(0, ndig))
A3 = np.loadtxt("train_dig3.txt", usecols=range(0, ndig))
A4 = np.loadtxt("train_dig4.txt", usecols=range(0, ndig))
A5 = np.loadtxt("train_dig5.txt", usecols=range(0, ndig))
A6 = np.loadtxt("train_dig6.txt", usecols=range(0, ndig))
A7 = np.loadtxt("train_dig7.txt", usecols=range(0, ndig))
A8 = np.loadtxt("train_dig8.txt", usecols=range(0, ndig))
A9 = np.loadtxt("train_dig9.txt", usecols=range(0, ndig))

p = 10
