#      _   _ _
#     | | | | |        Heloisa Lazari & Lucca Alipio
#     | |_| | |        Ciências Moleculares - USP
#     |  _  | |___     heloisalbento@usp.br | luccapagnan@usp.br
#     |_| |_|_____|
#
#     Primeira tarefa do EP de Computação III
#
#     Resolver Wx = b usando uma transformação tal que W_{nxm} vira R,
#     uma matriz triangular superior
#
###############################################################################

import math

import numpy as np

###############################################################################

#     Método de rotação de Givens disponibilizado no E-Disciplinas no arquivo
#     vector_oper.py


def Rotgivens(W, n, m, i, j, c, s):
    W[i, 0:m], W[j, 0:m] = c * W[i, 0:m] - s * W[j, 0:m], s * W[i, 0:m] + c * W[j, 0:m]
    return W


###############################################################################

# Função para verifiar se uma matriz é triangular superior


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

# Solução para a equação Wx = b


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


def Transformation(W, b):
    tsup = False
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
    tsup = Is_tsup(W, Wn, Wm)
    if tsup is False:
        mat = Transformation(W, b)
        W = mat[0]
        b = mat[1]
    return W, b


###############################################################################

# Cria a matriz, aleatoriamente ou não
# W = np.random.rand(n, m)
# W = np.empty((64, 64), dtype=float)
# for i in range(0, 64):
#     for j in range(0, 64):
#         if i == j:
#             W[i, j] = 2
#         elif abs(i - j) == 1:
#             W[i, j] = 1
#         else:
#             W[i, j] = 0
# b = np.ones((64, 1), dtype=float)
W = np.array([[3 / 5, 0], [0, 1], [4 / 5, 0]], dtype=float)
b = np.array([[3 / 10, 3 / 5, 0], [1 / 2, 0, 1], [4 / 10, 4 / 5, 0]], dtype=float)

# Para simplificar, usamos valores da matriz de rotacao quaisquer para
# teste mas garantimos que matriz seja ortogonal (para os testes de
# performace isso nao importa)

print("A matriz W original:")
print(W, end="\n")
print("A matriz b original:")
print(b, end="\n")

transf = Transformation(W, b)
W = transf[0]
b = transf[1]

print("A matriz após a rotação:")
print(W, end="\n")
print("O vetor b após a rotação:")
print(b, end="\n")


# Soluciona a equação Rx = b'e printa a matrix x resultante
x = Solution(W, b)
print("A matriz x:")
print(x)
