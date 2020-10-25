# Correspondente a parimeira tarefa do EP de Computação III

# Resolver Wx = b usando uma transformação tal que W_{nxm} vira R, uma
# matriz triangular superior

import numpy as np
import math

#########################################################################

# Método de rotação de Givens disponibilizado no E-Disciplinas no arquivo
# vector_oper.py
def Rotgivens(W,n,m,i,j,c,s):
  W[i,0:m] , W[j,0:m] = c * W[i,0:m] - s * W[j,0:m] , s * W[i,0:m] + c * W[j,0:m]
  return W

#########################################################################

# Função para verifiar se uma matriz é triangular superior
def Is_tsup(W):
    is_valid = True
    for collum in range(0, m):
        for line in range(collum + 1, n):
            if W[line][collum] != 0:
                is_valid = False
                return is_valid
    return is_valid

#########################################################################

#Tamanho da matriz W
n = 3
m = 3

# Cria a matriz, aleatoriamente ou não
# W = np.random.rand(n, m)
W = np.array([[6,5,0],[5,1,4],[0,4,3]])
vecaux = np.random.rand(m)

#Fazemos uma copia para guardar a original
A = W.copy()

# Para simplificar, usamos valores da matriz de rotacao quaisquer para
# teste mas garantimos que matriz seja ortogonal (para os testes de
# performace isso nao importa)
c = 0.5
s = math.sqrt(3.0)/2.0

print(W)

# Boolean que verifica se a matriz é triangular superior
tsup = False

# Loop que aplica RotGivens para todo elemento inferior a diagonal enquanto
# ela não é triangular superior
while tsup != True:
    for j in range(0, m):
        for i in range(j + 1, n):
            if W[i][j] != 0:
                W = Rotgivens(W,n,m,i,j,c,s)
    tsup = Is_tsup(W)
    #print(tsup)

print(W)
