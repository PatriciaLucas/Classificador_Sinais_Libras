import os
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import pandas as pd

def processing_rawdata(sinais, sinalizadores, gravacoes, path_data, path_save):
  for sinalizador in sinalizadores:
    num_sinalizador = sinalizadores.index(sinalizador)+1
    for sinal in sinais:
      for gravacao in gravacoes:
        arquivo = path_data + sinalizador + '/' + sinal + '/' + str(num_sinalizador) + '-' + sinal + '_' + gravacao + 'Body.txt'
        
        dadosBody = pd.read_csv(arquivo, header=None, delimiter=r"\s+") # lendo o arquivo
        dadosBody = pd.DataFrame.transpose(dadosBody)
        dadosBody17 = dadosBody.drop(dadosBody.index[[12,13,14,15,16,17,18,19]]) # excluindo pontos que foram inferidos, mas nao foram capturados pelo kinect
        dadosBody10 = dadosBody17.drop(dadosBody17.index[[0,1,2,3,4,8,12]]) # excluindo pontos que nao apresentaram movimento durante a execucao dos sinais

        # pegando os valores de x e y
        x = pd.DataFrame()
        y = pd.DataFrame()
        j=0
        for i in range(0, 1950,13):
          x = pd.concat([x, dadosBody10[i]], axis=1)
          y = pd.concat([y, dadosBody10[i+1]], axis=1)
          j+=1

        # ordenando os índices dos dataframes
        order1 = x.reset_index()
        order1 = order1.drop(columns="index")
        order2 = order1.T
        order2 = order2.reset_index()
        x = order2.drop(columns="index")
        x = x.T

        order1 = y.reset_index()
        order1 = order1.drop(columns="index")
        order2 = order1.T
        order2 = order2.reset_index()
        y = order2.drop(columns="index")
        y = y.T

        matriz = x.append(y) # matriz 20x150 com referência ao posicionamento da cabeca

        np.save(path_save + str(num_sinalizador) + '-' + sinal + '_' + gravacao + '.npy', matriz)
  return matriz

def processing_relationdata(sinais, sinalizadores, gravacoes, path_data, path_save):
  for sinalizador in sinalizadores:
    num_sinalizador = sinalizadores.index(sinalizador)+1
    for sinal in sinais:
      for gravacao in gravacoes:
        arquivo = path_data + sinalizador + '/' + sinal + '/' + str(num_sinalizador) + '-' + sinal + '_' + gravacao + 'Body.txt'
        
        dadosBody = pd.read_csv(arquivo, header=None, delimiter=r"\s+") # lendo o arquivo
        dadosBody = pd.DataFrame.transpose(dadosBody)
        dadosBody17 = dadosBody.drop(dadosBody.index[[12,13,14,15,16,17,18,19]]) # excluindo pontos que foram inferidos, mas nao foram capturados pelo kinect
        dadosBody10 = dadosBody17.drop(dadosBody17.index[[0,1,2,3,4,8,12]]) # excluindo pontos que nao apresentaram movimento durante a execucao dos sinais

        # pegando os valores de x e y
        x = pd.DataFrame()
        y = pd.DataFrame()
        j=0
        for i in range(0, 1950,13):
          x = pd.concat([x, dadosBody10[i]-dadosBody[i][3]], axis=1)
          y = pd.concat([y, dadosBody10[i+1]-dadosBody[i+1][3]], axis=1)
          j+=1

        # ordenando os índices dos dataframes
        order1 = x.reset_index()
        order1 = order1.drop(columns="index")
        order2 = order1.T
        order2 = order2.reset_index()
        x = order2.drop(columns="index")
        x = x.T

        order1 = y.reset_index()
        order1 = order1.drop(columns="index")
        order2 = order1.T
        order2 = order2.reset_index()
        y = order2.drop(columns="index")
        y = y.T

        matriz = x.append(y) # matriz 20x150 com referência ao posicionamento da cabeca

        np.save(path_save + str(num_sinalizador) + '-' + sinal + '_' + gravacao + '.npy', matriz)
  return matriz

def return_labels(matriz_path):
  lista = []
  labels = []
  matrizPaths = os.listdir(matriz_path) #nome dos arquivos
  for matrix in matrizPaths: #Exemplo: '10-15Maca_3.npy'
    label = ''.join(i for i in matrix if not i.isdigit()) #Exemplo: '-Maca_.npy'
    label = ''.join(c for c in label if c not in '-') #Exemplo: 'Maca_.npy'
    label = ''.join(c for c in label if c not in '_') #Exemplo: 'Maca.npy'
    label = label.replace('.npy', '') #Exemplo: 'Maca'
    labels.append(label) #adiciona o item no final da lista
    lista.append(matrix)

  lb = LabelBinarizer()
  dados_Y = lb.fit_transform(labels)
  indices = range(len(labels))
  return indices, dados_Y, lista

def generate(X, matriz_path, lista):
    """
    Gera dados de treino para a CNN4
    :parametro X: dados de entrada de treino
    :return: dados de entrada e saída de treino
    """
    data = []
    labels = []
    idx= X
    matriz_treino = map(lambda i: lista[i], idx)
    matriz_treino = sorted(matriz_treino, key=lambda x: (int(re.sub('\D','',x)),x))

    for matriz in matriz_treino:
      mat = np.load(matriz_path + '/' + matriz)
      label = ''.join(i for i in matriz if not i.isdigit()) #Exemplo: '-Maca_.npy'
      label = ''.join(c for c in label if c not in '-') #Exemplo: 'Maca_.npy'
      label = ''.join(c for c in label if c not in '_') #Exemplo: 'Maca.npy'
      label = label.replace('.npy', '') #Exemplo: 'Maca'
      labels.append(label)
      data.append(mat)
          
    x_train = np.array(data, dtype = 'float32')
    lb = LabelBinarizer()
    y_train = lb.fit_transform(labels)
    y_train = np.stack((y_train,)*1, axis=-1)
    y_train = y_train.squeeze().argmax(axis=1)
    return x_train, y_train

def rolling_average(matrix, lag):
  """
  Gera amostras de série temporais multivariadas usando Média Móvel
  :parametro matrix_x: matriz com todas as séries multivariadas
  :parametro lag: número de atrasos usados para a média móvel
  :return: série multivariadas suavizadas pela média móvel, índices das classes da nova matriz
  """
  sample = np.zeros((20, 150))
  matrix_samples = matrix
  for point in range(matrix.shape[0]):
      dataset = pd.DataFrame({'Point':matrix[point][:]})
      #dataset.index = pd.to_datetime(dataset.index)
      rolling = dataset.rolling(window=lag)
      rolling_mean = rolling.mean()
      for i in range(lag-1):
        rolling_mean['Point'][i] = dataset['Point'][i]
      rolling_mean_array = rolling_mean.to_numpy()
      rolling_mean_array = np.reshape(rolling_mean_array, (-1, 150))
      sample[point] = rolling_mean_array

  return sample


def generate_samples(matrix, lag):
  """
  Chama a função que gera amostras sintéticas de sinais de libras com média móvel
  :parametro matrix_x: matriz com todas as séries multivariadas
  :parametro lag: número de atrasos usados para a média móvel
  :return: matriz numpy com séries multivariadas suavizadas pela média móvel
  """
  new_matrix = np.zeros((matrix.shape[0],matrix.shape[1], matrix.shape[2]))
  for sample in range(matrix.shape[0]):
    new_sample = rolling_average(matrix[sample,:,:], lag).reshape(-1,20,150)
    new_matrix[sample] = new_sample
  return new_matrix

def generate_train_test(matrix_path):
  indices, dados_Y, lista = return_labels(matrix_path)
  (train_X, test_X, train_Y, test_Y) = train_test_split(indices,dados_Y,random_state=42,test_size=0.25,stratify=dados_Y)
  X_train, y_train = generate(train_X, matrix_path, lista)
  X_test, y_test = generate(test_X, matrix_path, lista)
  return X_train, y_train, X_test, y_test