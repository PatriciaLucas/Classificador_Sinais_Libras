from CSTSL import base
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

def generate_samples(X, Y, num_samples):
  """
  Chama a função que gera amostras sintéticas de sinais de libras
  :parametro X: dados de entrada reais
  :parametro Y: dados de saída reais
  :parametro num_samples: número de amostras de cada classe que o usuário deseja gerar
  :return: base de dados real + base de dados sintética
  """
  X_new = X
  Y_new = Y
  for i in range(num_samples):
    X_new, Y_new = samples(X_new, Y_new)
  return X_new, Y_new

def samples(X, Y):
  """
  Gera amostras sintéticas de sinais de libras usando o KDE (kernel density estimation)
  :parametro X: dados de entrada reais
  :parametro Y: dados de saída reais
  :return: base de dados real + base de dados sintética
  """
  X_new = X
  Y_new = Y
  for c in range(20):
    point_mean = np.zeros((1, 20, 150))
    classe_point = np.zeros(1, dtype = int)
    classe_point[0] = int(c)
    classe =np.where(Y[:]==c)
    matrix_group = np.array(X)[classe[0]]
    for j in range(20):
      for i in range(150):
        point = matrix_group[:,j,i]
        #band = grid(point)
        kde_point = KernelDensity(kernel='gaussian', bandwidth=0.11).fit(point.reshape((-1,1)))
        point_mean[0,j,i] = ((kde_point.sample(1)).mean())
    X_new = np.concatenate((X_new, point_mean), axis=0)
    Y_new = np.concatenate((Y_new, classe_point), axis=0)
  return X_new, Y_new
