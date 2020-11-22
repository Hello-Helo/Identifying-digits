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

import cProfile
import math
import pdb
import time

import numpy as np

###############################################################################


def Constants(W, j, k, i):
    if abs(W[i, k]) > abs(W[j, k]):
        t = -W[j, k] / W[i, k]
        c = 1 / np.sqrt(1 + t ** 2)
        s = c * t
    else:
        t = -W[i, k] / W[j, k]
        s = 1 / np.sqrt(1 + t ** 2)
        c = s * t
    return s, c


###############################################################################


def Solve(W, b):
    Wn = W.shape[0]
    Wm = W.shape[1]
    bm = b.shape[1]
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
    Wn = W.shape[0]
    Wm = W.shape[1]
    for j in range(0, Wm):
        s = np.sqrt((np.sum(W[0:Wn, j] ** 2)))
        W[0:Wn, j] = W[0:Wn, j] / s
    return W


###############################################################################


def Erro(A, W, H):
    erro = 0
    WH = np.matmul(W, H)
    Am = A.shape[1]
    for j in range(0, Am):
        erro = erro + np.sum((A[:, j] - WH[:, j]) ** 2)
    return erro


###############################################################################


def Image_processing(W, m):
    for j in range(0, m):
        W[0:784, j] = W[0:784, j] / 255.0
    return W


###############################################################################


def Train(A, p):
    An = A.shape[0]
    Am = A.shape[1]
    W = np.random.random(size=(An, p))
    H = np.zeros((p, Am), dtype=float)

    Aprime = np.copy(A)

    E = 10000
    Edif = 10000
    iterations = 0

    A = np.copy(Aprime)

    while Edif > 0.000001 and iterations < 100:
        W = Normalize(W)

        H = Solve(W, A)

        A = np.copy(Aprime)

        H = np.clip(H, 0, None)
        At = np.transpose(A).copy()
        Ht = np.transpose(H).copy()

        Wt = Solve(Ht, At)

        A = np.copy(Aprime)

        W = np.transpose(Wt).copy()
        W = np.clip(W, 0, None)
        Eprev = E
        # E = np.linalg.norm(np.matmul(W, H) - A) ** 2
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


print("Iniciando o IdentificadorDeDígitos.py", end="\n\n")


print("Lendo as imagens...")
start_time = time.time()

ndig = 1000
real = np.loadtxt("test_index.txt")

# Criando a array 3D que guardará os dados das imagens
Images = np.zeros((784, ndig, 10), dtype=float)
for i in range(0, 10):
    Images[:, :, i] = (
        np.loadtxt("train_dig" + str(i) + ".txt", usecols=range(0, ndig)) / 255.0
    )
# Images[:, :, 1] = np.loadtxt("train_dig1.txt", usecols=range(0, ndig))
# Images[:, :, 2] = np.loadtxt("train_dig2.txt", usecols=range(0, ndig))
# Images[:, :, 3] = np.loadtxt("train_dig3.txt", usecols=range(0, ndig))
# Images[:, :, 4] = np.loadtxt("train_dig4.txt", usecols=range(0, ndig))
# Images[:, :, 5] = np.loadtxt("train_dig5.txt", usecols=range(0, ndig))
# Images[:, :, 6] = np.loadtxt("train_dig6.txt", usecols=range(0, ndig))
# Images[:, :, 7] = np.loadtxt("train_dig7.txt", usecols=range(0, ndig))
# Images[:, :, 8] = np.loadtxt("train_dig8.txt", usecols=range(0, ndig))
# Images[:, :, 9] = np.loadtxt("train_dig9.txt", usecols=range(0, ndig))

# Processando as imagens
for i in range(0, 10):
    Images[:, :, i] = Image_processing(Images[:, :, i], ndig)

# Controle do tempo
t1 = time.time() - start_time
print("Feito em", t1, "segundos")
print("O tempo total é de", t1, "segundos", end="\n\n")


########################################################


print("Criando os parâmetros da AI...")

p = 10
W = np.zeros((784, p, 10), dtype=float)
for i in range(0, 10):
    W[:, :, i] = Train(Images[:, :, i], p)
    print("  Dígito", i, "treinado!")

t2 = time.time() - start_time
t12 = t2 - t1
print("Feito em", t12, "seconds")
print("O tempo total é de", t2, "segundos", end="\n\n")


########################################################


print("Analizando os dígitos escritos...")

n_test = 10000
Atest = np.loadtxt("test_images.txt", usecols=range(0, n_test))

# Wd * H = A
H = np.zeros((p, n_test, 10))
for i in range(0, 10):
    H[:, :, i] = Solve(W[:, :, i].copy(), Atest.copy())

t3 = time.time() - start_time
t23 = t3 - t2
print("Feito em", t23, "segundos")
print("O tempo total é de", t3, "segundos", end="\n\n")


########################################################


print("Comparando os dígitos escritos com o nosso banco de dados...")

D = np.zeros((784, n_test, 10))
for i in range(0, 10):
    D[:, :, i] = Difference(Atest, W[:, :, i], H[:, :, i])

t4 = time.time() - start_time
t34 = t4 - t3
print("Feito em", t34, "segundos")
print("O tempo total é de", t4, "segundos", end="\n\n")


########################################################


print("Identificando os dígitos...")

results = np.zeros((n_test), dtype=int)
error = np.zeros((n_test))

success = np.zeros((10))
bla = 0

for j in range(0, n_test):
    for i in range(0, 10):
        norm = np.sqrt(np.sum(D[:, j, i] ** 2))
        if i == 0:
            error[j] = norm
            results[j] = i
        else:
            if norm < error[j]:
                error[j] = norm
                results[j] = i

    if results[j] == real[j]:
        bla += 1

    # temp = results[j]
    # # print(temp)
    # if temp == real[j]:
    #     success[temp] = success[temp] + 1

t5 = time.time() - start_time
t45 = t5 - t4
print("Feito em", t45, "segundos")
print("O tempo total é de", t4, "segundos", end="\n\n")

print("Os resultados:")
# Sum = np.sum(success)
Percent = (bla / n_test) * 100
print("Conseguimos", bla, "dígitos indentificados corretamente")
print("Isso é", Percent, "%")
