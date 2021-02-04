import warnings
warnings.filterwarnings('ignore')
import numpy as np
import random
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, confusion_matrix

import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Flatten, SpatialDropout1D, Activation, Add, BatchNormalization, Conv1D, MaxPooling1D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from tcn import compiled_tcn

from CSTSL import base


def random_CNN():
  num_conv = random.randint(1, 9) # número de camadas convolucionais
  k = random.randint(0, 3) # tamanho do kernel de convolução
  pilhas = random.randint(1, 3) #número de pilhas [1,2,3]
  if k == 0: 
    kernel_size = 2
  elif k == 1:
    kernel_size = 3
  elif k == 2:
    kernel_size = 5
  else:
    kernel_size = 11
  
  return genotypeCNN(
    pilhas,
    random.randint(0, 2),   # número de filtros [16,32,64]
    random.uniform(0, 0.5),  # porcentagem dropout
    random.randint(0, 1),   # normalização (0 - sim, 1 - não)
    num_conv,
    kernel_size,
    [], #acc de teste do modelo
    []  #número de parâmetros do modelo
  )


def genotypeCNN(blocos, filters, dropout, norm, num_conv, kernel_size, acc, num_param):
  ind = {
      'pilhas': blocos, #número de blocos [1,2,3]
      'filters': filters, # número de filtros [16,32,64]
      'dropout': dropout, # porcentagem dropout [0...0.5]
      'norm': norm, # normalização (0 - sim, 1 - não)
      'num_conv': num_conv, # número de camadas convolucionais
      'kernel_size': kernel_size, # tamanho do kernel de convolução [2,3,5,11]
      'acc': acc, #acc de teste do modelo
      'num_param':num_param   #número de parâmetros do modelo

  }
  return ind

def modelo_CNN(X_train, y_train, individual, epocas):
    """
    Cria um modelo CNN4
    :parametro X_train: dados para treinamento
    :parametro y_train: rótulo dos dados de treinamento
    :parametro individual: dicionário com os hiperparâmetros do modelo
    :return: o modelo
    """
    warnings.filterwarnings('ignore')
    call = [EarlyStopping(monitor='loss', mode='min', patience=15, verbose=0),]
    if individual['filters'] == 0: 
        filters = 16
    elif individual['filters'] == 1:
        filters = 32
    else:
        filters = 64
    
    if individual['norm'] == 0:
        norm = False
    else:
        norm = True
  
    if individual['kernel_size'] == 0: 
        kernel_size = 2
    elif individual['kernel_size'] == 1:
        kernel_size = 3
    elif individual['kernel_size'] == 2:
        kernel_size = 5
    else:
        kernel_size = 11
    
    d = []
    for i in range(individual['num_conv']):
        d.append(2**i)
    
    model = compiled_tcn(return_sequences=False, num_feat=150, num_classes=20, nb_filters=filters, kernel_size=kernel_size, dilations=d,
                         padding='causal', dropout_rate=individual['dropout'], use_batch_norm=True, nb_stacks=individual['pilhas'], max_len=X_train[0:1].shape[1],
                         use_skip_connections=False)
    history = model.fit(X_train, y_train, epochs=epocas, callbacks = call, verbose=0)    
  
    return model, history

#-------------------------------------------------------------------------------------------------------------------------------------------------------------
def Ensemble(num_models):
    """
    Busca os hiperparâmetros dos modelos escolhidos para compor o ensemble
    :parametro series: base de dados
    :parametro n: tipo do modelo
    :parametro t: número modelos
    :return: lista com o dicionário dos modelos que irão compor o ensemble
    """
    models = []
    for i in range(num_models):
      models.append(random_CNN())
    return models
    
def fit(models, X_train, y_train, X_test, y_test, epocas):
    """
    Executa os modelos do ensemble
    :parametro models: modelos
    :parametro dataset: base de dados
    :return: yhat: valor previsto, y_test: valor real
    """
    yhats = []
    for i in range(len(models)):
        model, history = modelo_CNN(X_train, y_train, models[i], epocas)
        yhat = model.predict(X_test)
        yhat = yhat.squeeze().argmax(axis=1)
        yhats.append(yhat)
    return yhats
       
def evaluate(yhats, y_test):
    """
    Faz a previsão do ensemble usando a média de 100 amostras da distribuição de probabilidade
    :parametro kde_list: distribuição de probabilidade
    :parametro y_test: valores reais
    :return: rmse da previsão do ensemble e os valores previstos pelo ensemble
    """
    array_yhats = np.asarray(yhats)
    yhat_ensemble = stats.mode(array_yhats, axis=0)
    accuracy = accuracy_score(yhat_ensemble[0].reshape((-1,1)), y_test)
    tn, fp, fn, tp = confusion_matrix(yhat_ensemble[0].reshape((-1,1)), y_test)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    F1 = 2 * ((precision*recall) / (precision + recall))
    return accuracy, precision, recall, F1
