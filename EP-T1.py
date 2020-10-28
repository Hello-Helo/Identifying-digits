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

# public static void main(String args[])

#Tamanho da matriz W
n = 3
m = 3

# Cria a matriz, aleatoriamente ou não
# W = np.random.rand(n, m)
W = np.array([[6,5,0],[5,1,4],[0,4,3]])
b = np.array([[16],[19],[11]])

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
    for k in range(0, m):
        for j in range(n-1,k,-1):
            i = j-1
            if W[j,k] != 0:
                W = Rotgivens(W,n,m,i,j,c,s)
            if k == 0:
                b = Rotgivens(b,n,m,i,j,c,s)
    tsup = Is_tsup(W)

# Printa a matriz triangular superior
print('A matriz após a rotação:')
print(W, end = "\n")
print('O vetor b após a rotação:')
print(b)

x = np.array([[0],[0],[0]])
som = 0

for k in range (m-1, 0, -1):
	for j in range (k+1, m):
		som = W[k, j]*x[j, 0]
	x[k, 0] = (b[k, 0] - som)/W[k, k]

print(x)
