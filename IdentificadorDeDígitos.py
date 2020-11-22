#      _   _ _
#     | | | | |        Heloisa Lazari & Lucca Alipio
#     | |_| | |        Ciências Moleculares - USP
#     |  _  | |___     heloisalbento@usp.br | luccapagnan@usp.br
#     |_| |_|_____|
#
#     Terceira tarefa do EP de Computação III
#
#     Esse aquivo le o banco de dados e treia a AI para identificar os números
#     no arquivo test_images.txt
#
###############################################################################

import math
import time

import numpy as np

###############################################################################


# Cálculo das constantes para o RotGivens
def Constants(W, j, k, i):

    # Acha as constantes a partir da posição no vetor
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

    # Tamanho da matriz
    Am = A.shape[1]

    # Soma o quadrdo de todos os termos da diferença
    WH = np.matmul(W, H)
    erro = 0
    for j in range(0, Am):
        erro = erro + np.sum((A[:, j] - WH[:, j]) ** 2)

    # Retorna o erro
    return erro


###############################################################################


# Cria os parâmetros da AI para a identificação dos dígitos
def Train(A, p):

    # Tamanho e definição das matrizes (W será o parâmetro)
    An = A.shape[0]
    Am = A.shape[1]
    W = np.random.random(size=(An, p))
    H = np.zeros((p, Am), dtype=float)

    # Guarda uma cópia dos dados de cada um dos dígitos
    Aprime = np.copy(A)

    # Definições arbitrárias do erro e contagem de iterações para o loop while
    E = 10000
    Edif = 10000
    iterations = 0

    # Inicialicação do while para definição do parâmetro
    while Edif > 0.000001 and iterations < 100:

        # Normalização do pré-parâmetro
        W = Normalize(W)

        # Criação de uma matriz auxiliar positiva para refinamento do parâmetro
        H = Solve(W, A)
        H = np.clip(H, 0, None)

        # Volta dos dados origináis dos digitos
        A = np.copy(Aprime)

        # Transposição para o refinamento do parâmetro
        At = np.transpose(A).copy()
        Ht = np.transpose(H).copy()

        # Definindo um parâmetro mais refinado
        Wt = Solve(Ht, At)
        W = np.transpose(Wt).copy()
        W = np.clip(W, 0, None)

        # Volta dos dados origináis dos digitos
        A = np.copy(Aprime)

        # Calculo da diferença do erro entre os dois ultimos parâmetros
        Eprev = E
        E = Erro(A, W, H)
        Edif = abs(E - Eprev)

        # Mantem conhecimento do número refinamento do parâmetro
        iterations += 1

    # Retorna o parâmetro após cumprir os requerimentos (while)
    return W


###############################################################################


# Iniciando o programa
print("Iniciando o IdentificadorDeDígitos.py", end="\n\n")

# Os parâmetros do Machine Learning são escolhidos pelu usuário
ndig = int(input("Número de dígitos para o treinamento da AI (max. 4000): "))
p = int(input("Nível de precição da AI (max. 15): "))
print("\n")


#


# Primeira parte do processo - Ler as imagens para o Machine Learning
print("Lendo as imagens para o Machine Learning...")
start_time = time.time()

# Recebe uma array com os dígitos a serem identificados
n_test = 10000
Atest = np.loadtxt("dados_mnist/test_images.txt", usecols=range(0, n_test))

# Recebe uma array com a resposta dos dados a serem identificados para compa-
# ração dos resultados
real = np.loadtxt("dados_mnist/test_index.txt")

# Cria uma array 3D que guardará os dados das imagens para o treino da AI
Images = np.zeros((784, ndig, 10), dtype=float)
for i in range(0, 10):
    Images[:, :, i] = (
        np.loadtxt("dados_mnist/train_dig" + str(i) + ".txt", usecols=range(0, ndig))
        / 255.0
    )

# Controle do tempo da primeira parte
t1 = time.time() - start_time
print("  Feito em", t1, "segundos")
print("  O tempo total é de", t1, "segundos", end="\n\n")


#


# Segunda parte do processo - Treinar a AI
print("Criando os parâmetros da AI...")

# Treinamento para a criação do parâmetro de identificação dos dígitos
W = np.zeros((784, p, 10), dtype=float)
for i in range(0, 10):
    # inicio = time.time()
    W[:, :, i] = Train(Images[:, :, i], p)
    # fim = time.time()
    # print("    Dígito", i, "treinado em", fim - inicio, "segundos")
    print(i)

# Controle do tempo da segunda parte
t2 = time.time() - start_time
t12 = t2 - t1
print("  Feito em", t12, "seconds")
print("  O tempo total é de", t2, "segundos", end="\n\n")


#


# Terceira parte do processo - Comparar digitos escritos com os dados
print("Comparando os dígitos escritos com o nosso banco de dados...")

# Criação de um parâmetro para comparar os digitos escritos com o dados da AI
H = np.zeros((p, n_test, 10))
for i in range(0, 10):
    H[:, :, i] = Solve(W[:, :, i].copy(), Atest.copy())

# Achar a diferença dos parâmetros acima com os dados da AI
D = np.zeros((784, n_test, 10))
for i in range(0, 10):
    D[:, :, i] = np.subtract(Atest, np.matmul(W[:, :, i], H[:, :, i]))

# Controle do tempo da terceira parte
t3 = time.time() - start_time
t23 = t3 - t2
print("  Feito em", t23, "segundos")
print("  O tempo total é de", t3, "segundos", end="\n\n")


#


# Quarta parte do processo - Identificar os digitos
print("Identificando os dígitos...")

# Criar matrizes para guardar o dígito identificado e o erro com os dados da AI
results = np.zeros((n_test), dtype=int)
error = np.zeros((n_test))

# Criação dos locais para guardar o número de respostas corretas
success = np.zeros((10))
total = 0

# Array com o total de aparições reais de um determinado dígito
avl = np.zeros((10))

# Compara o resultado identificado com o real e guarda o resultado em results[]
for j in range(0, n_test):
    for i in range(0, 10):
        norm = math.sqrt(np.sum(D[:, j, i] ** 2))
        if i == 0:
            error[j] = norm
            results[j] = i
        else:
            if norm < error[j]:
                error[j] = norm
                results[j] = i

    # Guarda o número de aparições reais
    avl[int(real[j])] += 1

    # Conta o número de resultados corretos total e do dígito específico
    if results[j] == real[j]:
        total += 1
        success[int(real[j])] += 1

# Controle do tempo da terceira parte
t4 = time.time() - start_time
t34 = t4 - t3
print("  Feito em", t34, "segundos")
print("  O tempo total é de", t4, "segundos", end="\n\n")


#


# Quinta parte do processo - Os resultados
print("Os resultados gerais:")

# Calcula a porgentagem geral de acertos
Percent = (total / n_test) * 100
print("  Conseguimos", total, "dígitos indentificados corretamente")
print("  Isso é", Percent, "%")

# Calcula a porcentagem de acerto por digito
print("Resultados por dígito:")
for i in range(0, 10):
    print("  ", i, ": ", success[i] / avl[i] * 100, end="% \n")
