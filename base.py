import os
import numpy as np
import pandas as pd
from random import randint
from numpy import random

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

  return x,y

def norm01(x,y)
  # normalizacao 0-1
  xBody = x.describe().loc[['max']]
  maximoX = xBody.T.max()
  yBody = y.describe().loc[['max']]
  maximoY = yBody.T.max()

  xBody = x.describe().loc[['min']]
  minimoX = xBody.T.min()
  yBody = y.describe().loc[['min']]
  minimoY = yBody.T.min()

  return maximoX, maximoY, minimoX, minimoY

def matriznorm(x, y, maximoX, maximoY, minimoX, minimoY)
  df_normX = (x - float(minimoX)) / (float(maximoX) - float(minimoX))
  df_normY = (y - float(minimoY)) / (float(maximoY) - float(minimoY))

  df_normX,df_normY = order(df_normX,df_normY)

  matriz = df_normX.append(df_normY) 
  return matriz

def processing_normsigndata(sinais, sinalizadores, gravacoes, path_data, path_save):

  for sinal in sinais:
    dadossinal_X = pd.DataFrame()  
    dadossinal_Y = pd.DataFrame() 
    
    for sinalizador in sinalizadores:
      num_sinalizador = sinalizadores.index(sinalizador)+1

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
        
        dadossinal_X = pd.concat([dadossinal_X, x], axis=0)
        dadossinal_Y = pd.concat([dadossinal_Y, y], axis=0)

        dadossinal_X,dadossinal_Y = order(dadossinal_X,dadossinal_Y)
        maximoX, maximoY, minimoX, minimoY = norm01(dadossinal_Y,dadossinal_Y)
        matriz = matriznorm(x, y, maximoX, maximoY, minimoX, minimoY)

        np.save(path_save + str(num_sinalizador) + '-' + sinal + '_' + gravacao + '.npy', matriz)

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

        x,y = order(x,y)
        maximoX, maximoY, minimoX, minimoY = norm01(x,y)
        matriz = matriznorm(x, y, maximoX, maximoY, minimoX, minimoY)
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

        x,y = order(x,y)
        maximoX, maximoY, minimoX, minimoY = norm01(x,y)
        matriz = matriznorm(x, y, maximoX, maximoY, minimoX, minimoY)
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

        x,y = order(x,y)
        maximoX, maximoY, minimoX, minimoY = norm01(x,y)
        matriz = matriznorm(x, y, maximoX, maximoY, minimoX, minimoY)
        np.save(path_save + str(num_sinalizador) + '-' + sinal + '_' + gravacao + '.npy', matriz)
  return matriz

def window_warp(x):
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    window_ratio=0.01
    scales=[0, 20]
    x = np.swapaxes(x, 1, 2)
    warp_scales = np.random.choice(scales, x.shape[0])
    warp_size = np.ceil(window_ratio*x.shape[1]).astype(int)
    window_steps = np.arange(warp_size)
        
    window_starts = np.random.randint(low=1, high=x.shape[1]-warp_size-1, size=(x.shape[0])).astype(int)
    window_ends = (window_starts + warp_size).astype(int)
            
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            start_seg = pat[:window_starts[i],dim]
            window_seg = np.interp(np.linspace(0, warp_size-1, num=int(warp_size*warp_scales[i])), window_steps, pat[window_starts[i]:window_ends[i],dim])
            end_seg = pat[window_ends[i]:,dim]
            warped = np.concatenate((start_seg, window_seg, end_seg))                
            ret[i,:,dim] = np.interp(np.arange(x.shape[1]), np.linspace(0, x.shape[1]-1., num=warped.size), warped).T
    return ret[0,:,:].T

def processing_warpdata(matriz_path, path_save):
  """
  :parametro path_data: caminho dos dados que serao transfomados
  :parametro path_save: onde a matriz sera salva
  :return: 
  """

  matrizPaths = os.listdir(matriz_path)
  dados = []
  for mat in matrizPaths:
    matrix = np.load(matriz_path + mat)
    matriz = window_warp(np.stack((matrix,)*1, axis=0))
    np.save(path_save + mat, matriz)

  return matriz

def shift(X, periods): 
  result = np.zeros_like(X)
  for col in range(X.shape[0]):
    df = pd.DataFrame(X[col])
    df = df.shift(periods=periods,fill_value=X[col,0])
    array = df.to_numpy()
    result[col] = array.reshape((150))
  return result

def processing_shiftdata(matriz_path, path_save):
  """
  :parametro path_data: caminho dos dados que serao transfomados
  :parametro path_save: onde a matriz sera salva
  :return: 
  """

  matrizPaths = os.listdir(matriz_path)
  dados = []
  for mat in matrizPaths:
    matrix = np.load(matriz_path + mat)
    num = int(random.normal(loc=0, scale=5))
    matriz = shift(matrix,periods=num)
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

