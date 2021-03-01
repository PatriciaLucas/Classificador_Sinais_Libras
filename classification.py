import os
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import pandas as pd

def return_labels(matriz_path):
  lista = []
  labels = []
  numsinalizadores = []
  matrizPaths = os.listdir(matriz_path) #nome dos arquivos
  for matrix in matrizPaths: #Exemplo: '10-15Maca_3.npy'
    numsinalizadores.append(matrix.split('-')[0])
    label = ''.join(i for i in matrix if not i.isdigit()) #Exemplo: '-Maca_.npy'
    label = ''.join(c for c in label if c not in '-') #Exemplo: 'Maca_.npy'
    label = ''.join(c for c in label if c not in '_') #Exemplo: 'Maca.npy'
    label = label.replace('.npy', '') #Exemplo: 'Maca'
    labels.append(label) #adiciona o item no final da lista
    lista.append(matrix)

  lb = LabelBinarizer()
  dados_Y = lb.fit_transform(labels)
  indices = range(len(labels))
  return numsinalizadores, indices, dados_Y, lista

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

def generate_train_test(matriz_path,numsinalizador):
  datatest = []
  labeltest = []
  datatrain = []
  labeltrain = []

  lista_pontos = [1,2,4,5,6,8,11,12,14,15,16,18]

  matrizPaths = os.listdir(matriz_path) #nome dos arquivos
  for matriz in matrizPaths: #Exemplo: '10-15Maca_3.npy'
    mat = np.load(matriz_path + '/' + matriz)
    label = ''.join(i for i in matriz if not i.isdigit()) #Exemplo: '-Maca_.npy'
    label = ''.join(c for c in label if c not in '-') #Exemplo: 'Maca_.npy'
    label = ''.join(c for c in label if c not in '_') #Exemplo: 'Maca.npy'
    label = label.replace('.npy', '') #Exemplo: 'Maca'
    
    if (matriz.split('-')[0] == numsinalizador): 
      labeltest.append(label)
      datatest.append(mat[lista_pontos])
    else:    
      labeltrain.append(label)
      datatrain.append(mat[lista_pontos])

  lb = LabelBinarizer()
  y_train = lb.fit_transform(labeltrain)
  y_train = np.stack((y_train,)*1, axis=-1)
  y_train = y_train.squeeze().argmax(axis=1)

  y_test = lb.fit_transform(labeltest)
  y_test = np.stack((y_test,)*1, axis=-1)
  y_test = y_test.squeeze().argmax(axis=1)

  x_train = np.array(datatrain, dtype = 'float32')
  x_test = np.array(datatest, dtype = 'float32')

  return x_train, y_train, x_test, y_test
