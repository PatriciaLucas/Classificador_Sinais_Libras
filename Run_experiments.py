import numpy as np
import pandas as pd
import sqlite3
import contextlib
from sklearn.utils import shuffle
import random

#Função para executar INSERT INTO
def execute_insert(sql,data,database_path):
    """
    Função para executar INSERT INTO
    :parametro sql: string com código sql
    :parametro data: dados que serão inseridos no banco de dados
    :parametro database_path: caminho para o banco de dados
    """
    with contextlib.closing(sqlite3.connect(database_path)) as conn: 
        with conn:
            with contextlib.closing(conn.cursor()) as cursor: 
                cursor.execute(sql,data)
                return cursor.fetchall()


def execute(sql,database_path):
    """
    Função para executar CREATE TABLE
    :parametro sql: string com código sql
    :parametro database_path: caminho para o banco de dados
    :return: dataframe com os valores retornados pela consulta sql
    """
    with contextlib.closing(sqlite3.connect(database_path)) as conn:
        with conn: 
            with contextlib.closing(conn.cursor()) as cursor: 
                cursor.execute(sql)
                return cursor.fetchall()

def train_test_split_sinalizador(matriz, sinais, sinalizadores, sinalizador):
    index1 = np.where(sinalizadores == str(sinalizador))
    index2 = np.where(sinalizadores != str(sinalizador))
    X_train = np.delete(matriz, index1, axis=0)
    y_train = np.delete(sinais, index1)
    X_test = np.delete(matriz, index2, axis=0)
    y_test = np.delete(sinais, index2)
    return X_train, X_test, y_train, y_test

def experiment(matriz, sinais, sinalizadores, name_experiment, num_execute, database_path, num_feat, num_classes, nb_filters, kernel_size, dilations, dropout_rate, nb_stacks, num_sinalizadores=12, form='sinalizador'):
  from sklearn.model_selection import train_test_split
  import time
  execute("CREATE TABLE IF NOT EXISTS results(experiment TEXT, sinalizador TEXT, accuracy FLOAT, precision FLOAT, recall FLOAT, f1 FLOAT, tempo FLOAT, y_hat BLOB, y_test BLOB)",database_path)
  list_sinalizadores = sorted(random.sample(np.arange(1,12+1).tolist(),num_sinalizadores))
  for exec in range(num_execute):
      if form == 'sinalizador':
        for sinalizador in list_sinalizadores:
            X_train, X_test, y_train, y_test = train_test_split_sinalizador(matriz, sinais, sinalizadores, sinalizador)
            start_time = time.time()
            X_train, y_train = shuffle(X_train, y_train)
            X_test, y_test = shuffle(X_test, y_test)
            print("Execução:", exec, " Sinalizador:", sinalizador)
            accuracy, precision, recall, f1, yhat, y_test = individual(X_train, y_train, X_test, y_test, num_feat, num_classes, nb_filters, 
                                                                       kernel_size, dilations, dropout_rate, nb_stacks)
            tempo = time.time() - start_time
            execute_insert("INSERT INTO results VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",(name_experiment, sinalizador, accuracy, precision, recall, 
                                                                           f1, tempo, yhat.tostring(), y_test.tostring()),database_path)
      else:
        l = []
        X_train, X_test, y_train, y_test = train_test_split(matriz, sinais, test_size=0.025)
        X_train, y_train = shuffle(X_train, y_train)
        X_test, y_test = shuffle(X_test, y_test)
        X_train = np.concatenate((X_train, X_test[:20]), axis=0)
        y_train = np.concatenate((y_train, y_test[:20]), axis=0)
        X_test = X_test[20:]
        y_test = y_test[20:]
        start_time = time.time()
        print("Execução: ", exec)
        accuracy, precision, recall, f1, yhat, y_test = individual(X_train, y_train, X_test, y_test, num_feat, num_classes, nb_filters, 
                                                                   kernel_size, dilations, dropout_rate, nb_stacks)
        tempo = time.time() - start_time
        execute_insert("INSERT INTO results VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",(name_experiment, '-', accuracy, precision, recall, 
                                                                           f1, tempo, yhat.tostring(), y_test.tostring()),database_path)
  return

def individual(X_train, y_train, X_test, y_test, num_feat, num_classes, nb_filters, kernel_size, dilations, dropout_rate, nb_stacks):
  from keras.callbacks import EarlyStopping  
  from tcn import compiled_tcn
  from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
  call = [EarlyStopping(monitor='loss', mode='min', patience=15, verbose=1),]
  model = compiled_tcn(return_sequences=False,num_feat=num_feat,num_classes=num_classes,nb_filters=nb_filters,kernel_size=kernel_size,dilations=[2 ** i for i in range(dilations)],
                       padding='causal',dropout_rate=dropout_rate,use_batch_norm=True,nb_stacks=nb_stacks,max_len=X_train[0:1].shape[1],opt='adam',
                       use_skip_connections=True)

  history = model.fit(X_train, y_train, epochs=100, workers=4, use_multiprocessing=True, callbacks = call, verbose=0)
  yhat = model.predict(X_test).squeeze().argmax(axis=1)
  accuracy = model.evaluate(X_test, y_test)[1]
  precision = precision_score(y_test, yhat, average='macro')
  recall = recall_score(y_test, yhat, average='macro')
  f1 = f1_score(y_test, yhat, average='macro')
  return accuracy, precision, recall, f1, yhat, y_test
