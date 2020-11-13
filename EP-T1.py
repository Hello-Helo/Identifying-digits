# Correspondente a parimeira tarefa do EP de Computação III

# Resolver Wx = b usando uma transformação tal que W_{nxm} vira R, uma
# matriz triangular superior

import math

import numpy as np

#########################################################################

# Método de rotação de Givens disponibilizado no E-Disciplinas no arquivo
# vector_oper.py


def Rotgivens(W, n, m, i, j, c, s):
    W[i, 0:m], W[j, 0:m] = c * W[i, 0:m] - s * W[j, 0:m], s * W[i, 0:m] + c * W[j, 0:m]
    return W


#########################################################################

# Função para verifiar se uma matriz é triangular superior


def Is_tsup(W, Wn, Wm):
    is_valid = True
    for collum in range(0, Wm):
        for line in range(collum + 1, Wn):
            if W[line][collum] != 0:
                is_valid = False
                return is_valid
    return is_valid


#########################################################################


def Constants(W, j, k):
    if abs(W[i, k]) > abs(W[j, k]):
        t = -W[j, k] / W[i, k]
        c = 1 / math.sqrt(1 + t ** 2)
        s = c * t
    else:
        t = -W[i, k] / W[j, k]
        s = 1 / math.sqrt(1 + t ** 2)
        c = s * t
    return s, c


#########################################################################

# Solução para a equação Wx = b


def Solution(W, b):
    # Tmanho das matrizes
    Wn = np.atleast_2d(W).shape[0]
    Wm = np.atleast_2d(W).shape[1]
    bm = np.atleast_2d(b).shape[1]
    # Solução
    x = np.empty((Wn, bm), dtype=float)
    for w in range(0, bm):
        for k in range(Wm - 1, -1, -1):
            som = 0
            for j in range(k + 1, Wm):
                som = som + W[k, j] * x[j, w]
            x[k, w] = (b[k, w] - som) / W[k, k]
    return x


#########################################################################

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
W = np.array([[6, 5, 0], [5, 1, 4], [0, 4, 1]], dtype=float)
b = np.array([[16], [19], [11]], dtype=float)

# Tamanho da matriz W
Wn = np.atleast_2d(W).shape[0]
Wm = np.atleast_2d(W).shape[1]

# Tamanho da matriz b
bn = np.atleast_2d(b).shape[0]
bm = np.atleast_2d(b).shape[1]

# Para simplificar, usamos valores da matriz de rotacao quaisquer para
# teste mas garantimos que matriz seja ortogonal (para os testes de
# performace isso nao importa)
c = 0.5
s = math.sqrt(3.0) / 2.0
t = 0

print("A matriz W original:")
print(W, end="\n")
print("A matriz b original:")
print(b, end="\n")

# Boolean que verifica se a matriz é triangular superior
tsup = False

# Loop que aplica RotGivens para todo elemento inferior a diagonal enquanto
# ela não é triangular superior
while tsup is False:
    for k in range(0, Wm):
        for j in range(Wn - 1, k, -1):
            i = j - 1
            if W[j, k] != 0:
                const = Constants(W, j, k)
                s = const[0]
                c = const[1]
                W = Rotgivens(W, Wn, Wm, i, j, c, s)
                b = Rotgivens(b, Wn, Wm, i, j, c, s)
    tsup = Is_tsup(W, Wn, Wm)


print("A matriz após a rotação:")
print(W, end="\n")
print("O vetor b após a rotação:")
print(b, end="\n")


# Soluciona a equação Rx = b'e printa a matrix x resultante
x = Solution(W, b)
print("A matriz após a rotação:")
print(x)
