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
    # Tamanho das matrizes
    Wm = W.shape[1]
    bm = b.shape[1]

    # Acha a solução
    x = np.zeros((Wm, bm), dtype=float)
    for w in range(0, bm):
        for k in range(Wm - 1, -1, -1):
            som = np.sum(W[k, k + 1 :] * x[k + 1 :, w])
            x[k, w] = (b[k, w] - som) / W[k, k]

    # Retorna a solução
    return x


###############################################################################


def Transformation(W, b):
    # Tamanho das matrizes
    Wn = W.shape[0]
    Wm = W.shape[1]
    bm = b.shape[1]

    # Aplica RotGivens nas matrizes W e b
    for k in range(0, Wm):
        for j in range(Wn - 1, k, -1):
            i = j - 1
            if W[j, k] != 0:
                const = Constants(W, j, k, i)
                s = const[0]
                c = const[1]
                W[i, 0:Wm], W[j, 0:Wm] = (
                    c * W[i, 0:Wm] - s * W[j, 0:Wm],
                    s * W[i, 0:Wm] + c * W[j, 0:Wm],
                )
                b[i, 0:bm], b[j, 0:bm] = (
                    c * b[i, 0:bm] - s * b[j, 0:bm],
                    s * b[i, 0:bm] + c * b[j, 0:bm],
                )
    return W, b


###############################################################################

c = False
while c is False:
    v = input("Escolha o exemplo da Tarefa 1 a ser utilizado (a, b, c ou d): ")
    if v == "a":
        W = np.empty((64, 64), dtype=float)
        for i in range(0, 64):
            for j in range(0, 64):
                if i == j:
                    W[i, j] = 2
                elif abs(i - j) == 1:
                    W[i, j] = 1
                else:
                    W[i, j] = 0
        b = np.ones((64, 1), dtype=float)
        c = True
    elif v == "b":
        W = np.empty((20, 17))
        for i in range(0, 20):
            for j in range(0, 17):
                if abs(i - j) > 4:
                    W[i, j] = 0
                else:
                    W[i, j] = 1 / (i + j + 1)
        b = np.ones((20, 1), dtype=float)
        c = True
    elif v == "c":
        W = np.empty((64, 64), dtype=float)
        for i in range(0, 64):
            for j in range(0, 64):
                if i == j:
                    W[i, j] = 2
                elif abs(i - j) == 1:
                    W[i, j] = 1
                else:
                    W[i, j] = 0
        b = np.array([(1, i, 2 * i - 1) for i in range(1, 65)], dtype=float)
        c = True
    elif v == "d":
        W = np.empty((20, 17))
        for i in range(0, 20):
            for j in range(0, 17):
                if abs(i - j) > 4:
                    W[i, j] = 0
                else:
                    W[i, j] = 1 / (i + j + 1)
        b = np.array([(1, i, 2 * i - 1) for i in range(1, 21)], dtype=float)
        c = True
    else:
        print("Opção inválida", end="\n\n")

Wc = np.copy(W)

# Cria a matriz, aleatoriamente ou não
# W = np.random.rand(n, m)
# W = np.array([[1, 2], [2, 1], [1, 2]], dtype=float)
# b = np.array([[5, 4, 5], [4, 5, 4], [5, 4, 5]], dtype=float)

# Para simplificar, usamos valores da matriz de rotacao quaisquer para
# teste mas garantimos que matriz seja ortogonal (para os testes de
# performace isso nao importa)

print("A matriz W original:")
print(W, end="\n\n")
print("A matriz b original:")
print(b, end="\n\n")

transf = Transformation(W, b)
W = transf[0]
b = transf[1]

print("A matriz b rotação:")
print(b, end="\n\n")

# Soluciona a equação Rx = b'e printa a matrix x resultante
x = Solution(W, b)
print("A matriz x:")
print(x, end="\n\n")

print("A matriz W*x:")
print(np.matmul(Wc, x))
