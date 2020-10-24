# Correspondente a parimeira tarefa do EP de Computação III

import NumPy as np
import Matplotlib as mp

# Resolver Wx = b usando uma transformação tal que W_{nxm} vira R, uma
# matriz triangular superior

#########################################################################

# Método de rotação de Givens disponibilizado no E-Disciplinas no arquivo
# vector_oper.py
def Rotgivens(W,vecaux,n,m,i,j,c,s):
  vecaux[0:m] = c * W[i,0:m] - s * W[j,0:m]
  W[j,0:m] = s * W[i,0:m] + c * W[j,0:m]
  W[i,0:m] = vecaux[0:m]
  return W

#########################################################################

#Tamanho da matriz W
n = 784
m = 100000

# Preenche com qualquer coisa
W = np.random.rand(n, m)
vecaux = np.random.rand(m)

#Fazemos uma copia para guardar a original
A = W.copy()

#Vamos aplicar Givens nessas linhas
i = 4
j = 8
