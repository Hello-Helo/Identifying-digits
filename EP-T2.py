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


# Resolve um sistema de equações qualquer com RotGivens
def Solve(W, b):

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


# Aplica a normalização para cada coluna individualmente
def Normalize(W):

    # Tamanho das matrizes
    Wn = W.shape[0]
    Wm = W.shape[1]

    # Cada coluna tem seus valores divididos pelo quadrado da soma de todos os
    # termos
    for j in range(0, Wm):
        s = math.sqrt((np.sum(W[0:Wn, j] ** 2)))
        W[0:Wn, j] = W[0:Wn, j] / s

    # Retorna a matriz normalizada
    return W


###############################################################################


# Calcula o erro quadrado entre 3 matrizes
def Erro(A, W, H):

    # Inicio do cálculo
    erro = 0
    WH = np.matmul(W, H)

    # Tamanho da matriz
    Am = A.shape[1]

    # Soma o quadrdo de todos os termos da diferença
    for j in range(0, Am):
        erro = erro + np.sum((A[:, j] - WH[:, j]) ** 2)
    return erro


###############################################################################


A = np.array([[3 / 10, 3 / 5, 0], [1 / 2, 0, 1], [4 / 10, 4 / 5, 0]])

print("A matriz A:")
print(A, end="\n\n")

# An = Wn
# Am = Hm
# Wm = An - Arbitrario

p = 2

# Tamanho e definição das matrizes (W será aleatório)
An = A.shape[0]
Am = A.shape[1]
W = np.random.random(size=(An, p))
H = np.zeros((p, Am), dtype=float)

# Guarda uma cópia da matriz original
Aprime = np.copy(A)

# Definições arbitrárias do erro e contagem de iterações para o loop while
E = 10000
Edif = 10000
iterations = 0

# Inicialicação do while para definição de W
while Edif > 0.000001 and iterations < 100:

    # Normalização do W
    W = Normalize(W)

    # Refinando H
    H = Solve(W, A)
    H = np.clip(H, 0, None)

    # Volta da matriz original
    A = np.copy(Aprime)

    # Transposição para o refinamento de W
    At = np.transpose(A).copy()
    Ht = np.transpose(H).copy()

    # Definindo um W mais refinado
    Wt = Solve(Ht, At)
    W = np.transpose(Wt).copy()
    W = np.clip(W, 0, None)

    # Volta dos dados origináis
    A = np.copy(Aprime)

    # Calculo da diferença do erro entre os dois ultimos Ws
    Eprev = E
    E = Erro(A, W, H)
    Edif = abs(E - Eprev)

    # Mantem conhecimento do número refinamento de W
    iterations += 1

print("A matriz W encontrada:")
print(W, end="\n\n")

print("A matriz H encontrada:")
print(H, end="\n\n")

print("A matriz A a partir de WH:")
print(np.matmul(W, H), end="\n\n")

print("O erro em relação a A original")
print(E, end="\n\n")
