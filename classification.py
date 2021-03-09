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

def generate_train_test(matriz_path, form='sinalizador', numsinalizador='1', lista_sinalizadores):
  datatest = []
  labeltest = []
  datatrain = []
  labeltrain = []

  if form == 'sinalizador':
      lista_pontos = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]

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
          if matriz.split('-')[0] in lista_sinalizadores:
            labeltrain.append(label)
            datatrain.append(mat[lista_pontos])

      lb = LabelBinarizer()
      y_train = lb.fit_transform(labeltrain)
      y_train = np.stack((y_train,)*1, axis=-1)
      y_train = y_train.squeeze().argmax(axis=1)

      y_test = lb.fit_transform(labeltest)
      y_test = np.stack((y_test,)*1, axis=-1)
      y_test = y_test.squeeze().argmax(axis=1)

      X_train = np.array(datatrain, dtype = 'float32')
      X_test = np.array(datatest, dtype = 'float32')
  else:
      _,indices, dados_Y, lista = return_labels(matriz_path)
      (train_X, test_X, train_Y, test_Y) = train_test_split(indices,dados_Y,test_size=100,stratify=dados_Y,random_state=42)
      X_train, y_train = generate(train_X, matriz_path, lista)
      X_test, y_test = generate(test_X, matriz_path, lista)

  return X_train, y_train, X_test, y_test
