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
import time

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
        iterations += 1

    return W


###############################################################################


def Difference(A, W, H):
    WH = np.matmul(W, H)
    D = np.subtract(A, WH)
    return D


###############################################################################


def Norma(W, j):
    sum = 0
    for i in range(0, 784):
        sum = sum + W[i, j] ** 2
    norm = math.sqrt(sum)
    return norm


###############################################################################

start_time = time.time()
ndig = 100

T0 = np.loadtxt("train_dig0.txt", usecols=range(0, ndig))
T1 = np.loadtxt("train_dig1.txt", usecols=range(0, ndig))
T2 = np.loadtxt("train_dig2.txt", usecols=range(0, ndig))
T3 = np.loadtxt("train_dig3.txt", usecols=range(0, ndig))
T4 = np.loadtxt("train_dig4.txt", usecols=range(0, ndig))
T5 = np.loadtxt("train_dig5.txt", usecols=range(0, ndig))
T6 = np.loadtxt("train_dig6.txt", usecols=range(0, ndig))
T7 = np.loadtxt("train_dig7.txt", usecols=range(0, ndig))
T8 = np.loadtxt("train_dig8.txt", usecols=range(0, ndig))
T9 = np.loadtxt("train_dig9.txt", usecols=range(0, ndig))

t1 = time.time() - start_time
print("Reading training files - Done in ", t1)


p = 10

W0 = Train(T0, p)
W1 = Train(T1, p)
W2 = Train(T2, p)
W3 = Train(T3, p)
W4 = Train(T4, p)
W5 = Train(T5, p)
W6 = Train(T6, p)
W7 = Train(T7, p)
W8 = Train(T8, p)
W9 = Train(T9, p)

t2 = time.time() - start_time
print("Training the AI - Done in ", t2)

n_test = 10000
Atest = np.loadtxt("test_images.txt", usecols=range(0, n_test))

# Wd * H = A
WW0, A0 = Transformation(W0, Atest)
WW1, A1 = Transformation(W1, Atest)
WW2, A2 = Transformation(W2, Atest)
WW3, A3 = Transformation(W3, Atest)
WW4, A4 = Transformation(W4, Atest)
WW5, A5 = Transformation(W5, Atest)
WW6, A6 = Transformation(W6, Atest)
WW7, A7 = Transformation(W7, Atest)
WW8, A8 = Transformation(W8, Atest)
WW9, A9 = Transformation(W9, Atest)

H0 = Solution(WW0, A0)
H1 = Solution(WW1, A1)
H2 = Solution(WW2, A2)
H3 = Solution(WW3, A3)
H4 = Solution(WW4, A4)
H5 = Solution(WW5, A5)
H6 = Solution(WW6, A6)
H7 = Solution(WW7, A7)
H8 = Solution(WW8, A8)
H9 = Solution(WW9, A9)

t3 = time.time() - start_time
print("Solving the equations - Done in ", t3)

D0 = Difference(Atest, W0, H0)
D1 = Difference(Atest, W1, H1)
D2 = Difference(Atest, W2, H2)
D3 = Difference(Atest, W3, H3)
D4 = Difference(Atest, W4, H4)
D5 = Difference(Atest, W5, H5)
D6 = Difference(Atest, W6, H6)
D7 = Difference(Atest, W7, H7)
D8 = Difference(Atest, W8, H8)
D9 = Difference(Atest, W9, H9)

t4 = time.time() - start_time
print("Analizing the similarities - Done in ", t4)

results = np.empty((n_test))
error = np.empty((n_test))

for j in range(0, n_test):
    Norms = np.empty((10))
    Norms[0] = Norma(D0, j)
    Norms[1] = Norma(D1, j)
    Norms[2] = Norma(D2, j)
    Norms[3] = Norma(D3, j)
    Norms[4] = Norma(D4, j)
    Norms[5] = Norma(D5, j)
    Norms[6] = Norma(D6, j)
    Norms[7] = Norma(D7, j)
    Norms[8] = Norma(D8, j)
    Norms[9] = Norma(D9, j)

    index = np.argmin(Norms)
    results[j] = index
    error[j] = Norms[index]
