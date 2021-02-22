import os
import numpy as np
import pandas as pd
from random import randint

def order(x,y):
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

  del(order1)
  del(order2)

  # normalizacao 0-1
  xBody = x.describe().loc[['max']]
  maximoX = xBody.T.max()
  yBody = y.describe().loc[['max']]
  maximoY = yBody.T.max()

  xBody = x.describe().loc[['min']]
  minimoX = xBody.T.min()
  yBody = y.describe().loc[['min']]
  minimoY = yBody.T.min()

  df_normX = (x - float(minimoX)) / (float(maximoX) - float(minimoX))
  df_normY = (y - float(minimoY)) / (float(maximoY) - float(minimoY))

  del(xBody)
  del(yBody)
  del(maximoX)
  del(maximoY)
  del(minimoX)
  del(minimoY)

  matriz = df_normX.append(df_normY) 
  return matriz

def processing_rawdata(sinais, sinalizadores, gravacoes, path_data, path_save):
  """
  Gera amostras de série temporais multivariadas com as coordenadas dos pontos (dados brutos)
  :parametro sinais, sinalizadores e gravacoes: formam o nome dos arquivos que serao utilizados  
  :parametro path_data: caminho dos dados que serao transfomados
  :parametro path_save: onde a matriz sera salva
  :return: série multivariadas dos dados brutos
  """
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

        matriz = order(x,y)
        np.save(path_save + str(num_sinalizador) + '-' + sinal + '_' + gravacao + '.npy', matriz)
  return matriz

def processing_relationdata(sinais, sinalizadores, gravacoes, path_data, path_save):
  """
  Gera amostras de série temporais multivariadas com as coordenadas dos pontos relativos ao ponto da cabeca
  :parametro sinais, sinalizadores e gravacoes: formam o nome dos arquivos que serao utilizados  
  :parametro path_data: caminho dos dados que serao transfomados
  :parametro path_save: onde a matriz sera salva
  :return: série multivariadas dos dados relativos
  """
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
          x = pd.concat([x, dadosBody10[i]-dadosBody[i][3]], axis=1) # dadosBody[i][3]: ponto relativo a cabeca
          y = pd.concat([y, dadosBody10[i+1]-dadosBody[i+1][3]], axis=1)
          j+=1

        matriz = order(x,y)
        np.save(path_save + str(num_sinalizador) + '-' + sinal + '_' + gravacao + '.npy', matriz)
  return matriz

def processing_noisedata(sinais, sinalizadores, gravacoes, path_data, path_save):
  """
  Gera amostras de série temporais multivariadas com as coordenadas dos pontos relativos ao ponto da cabeca, sendo que o ponto de referência recebe um ruido gaussiano de desvio 0.05
  :parametro sinais, sinalizadores e gravacoes: formam o nome dos arquivos que serao utilizados  
  :parametro path_data: caminho dos dados que serao transfomados
  :parametro path_save: onde a matriz sera salva
  :return: série multivariadas dos dados relativos a coordenada da cabeça pertubada por um ruido
  """

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
          # np.random.normal(loc=dadosBody[i][3], scale=0.05, size=1): loc=valor médio, scale=desviopadrao, size=numero de pontos
          x = pd.concat([x, dadosBody10[i]-(dadosBody[i][3]+np.random.normal(loc=dadosBody[i][3], scale=0.05, size=1))], axis=1) # dadosBody[i][3]: ponto relativo a cabeca
          y = pd.concat([y, dadosBody10[i+1]-(dadosBody[i+1][3]+np.random.normal(loc=dadosBody[i+1][3], scale=0.05, size=1))], axis=1)
          j+=1

        matriz = order(x,y)
        np.save(path_save + str(num_sinalizador) + '-' + sinal + '_' + gravacao + '.npy', matriz)
  return matriz

def shift(X, periods): 
  result = np.zeros_like(X)
  for col in range(X.shape[0]):
    df = pd.DataFrame(X[col])
    df = df.shift(periods=periods,fill_value=X[col,0])
    array = df.to_numpy()
    result[col] = array.reshape((150))
  return result

def processing_shiftdata(matriz_path, path_save, list_shift):
  """
  :parametro path_data: caminho dos dados que serao transfomados
  :parametro path_save: onde a matriz sera salva
  :return: 
  """

  matrizPaths = os.listdir(matriz_path)
  dados = []
  for mat in matrizPaths:
    matrix = np.load(matriz_path + mat)
    num = randint(0,len(list_shift)-1)
    matriz = shift(matrix,periods=list_shift[num])
    np.save(path_save + mat, matriz)

  return matriz

def processing_smoothingdata(matriz_path, path_save, lag):
  """
  Chama a função que gera amostras sintéticas de sinais de libras com média móvel
  :parametro matriz_path: caminho das matrizes com todas as séries multivariadas
  :parametro path_save: onde a matriz sera salva
  :parametro lag: número de atrasos usados para a média móvel
  :return: matriz numpy com séries multivariadas suavizadas pela média móvel
  """

  matrizPaths = os.listdir(matriz_path)
  dados = []
  for mat in matrizPaths:
    matrix = np.load(matriz_path + '/' + mat)
    new_sample = rolling_average(matrix, lag)
    np.save(path_save + '/' + mat, new_sample)
    dados.append(new_sample)
    
  return dados


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

