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

def Is_tsup(W, Wn, Wm):
    is_valid = True
    for collum in range(0, Wm):
        for line in range(collum + 1, Wn):
            if W[line][collum] != 0:
                is_valid = False
                return is_valid
    return is_valid

#########################################################################

# public static void main(String args[])


# Cria a matriz, aleatoriamente ou não
# W = np.random.rand(n, m)
W = np.array([[6,5,0],[5,1,4],[0,4,3]])
b = np.array([[16],[19],[11]])

#Tamanho da matriz W
Wn = np.atleast_2d(W).shape[0]
Wm = np.atleast_2d(W).shape[1]

#Tamanho da matriz b
bn = np.atleast_2d(b).shape[0]
bm = np.atleast_2d(b).shape[1]

# Para simplificar, usamos valores da matriz de rotacao quaisquer para
# teste mas garantimos que matriz seja ortogonal (para os testes de
# performace isso nao importa)
c = 0.5
s = math.sqrt(3.0)/2.0

# Printa a matriz inicial
print('A matriz W original:')
print(W, end = "\n")
print('A matriz b original:')
print(b, end = "\n")

# Boolean que verifica se a matriz é triangular superior
tsup = False

# Loop que aplica RotGivens para todo elemento inferior a diagonal enquanto
# ela não é triangular superior
while tsup != True:
    for k in range(0, Wm):
        for j in range(Wn-1,k,-1):
            i = j-1
            if W[j,k] != 0:
                W = Rotgivens(W,Wn,Wm,i,j,c,s)
                if k < bm:
                    b = Rotgivens(b,Wn,Wm,i,j,c,s)
    tsup = Is_tsup(W, Wn, Wm)

# Printa a matriz triangular superior
print('A matriz após a rotação:')
print(W, end = "\n")
print('O vetor b após a rotação:')
print(b)
