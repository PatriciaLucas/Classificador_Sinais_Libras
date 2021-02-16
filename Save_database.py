import sqlite3
import contextlib
from CSTSL import classification
import numpy as np
from sklearn.utils import shuffle

#Função para executar INSERT INTO
def execute_insert(sql,data,database_path):
    """
    Função para executar INSERT INTO
    :parametro sql: string com código sql
    :parametro data: dados que serão inseridos no banco de dados
    :parametro database_path: caminho para o banco de dados
    """
    with contextlib.closing(sqlite3.connect(database_path)) as conn: # auto-closes
        with conn: # auto-commits
            with contextlib.closing(conn.cursor()) as cursor: # auto-closes
                cursor.execute(sql,data)
                return cursor.fetchall()


def execute(sql,database_path):
    """
    Função para executar INSERT INTO
    :parametro sql: string com código sql
    :parametro database_path: caminho para o banco de dados
    :return: dataframe com os valores retornados pela consulta sql
    """
    with contextlib.closing(sqlite3.connect(database_path)) as conn: # auto-closes
        with conn: # auto-commits
            with contextlib.closing(conn.cursor()) as cursor: # auto-closes
                cursor.execute(sql)
                return cursor.fetchall()

def concatenate_samples(X_train1, y_train1, X_test1, y_test1, X_train2, y_train2, X_test2, y_test2):
  X_train = np.concatenate((X_train1, X_train2), axis=0)
  y_train = np.concatenate((y_train1,y_train2), axis=0)
  X_test = np.concatenate((X_test1,X_test2), axis=0)
  y_test = np.concatenate((y_test1,y_test2), axis=0)
  return X_train, y_train, X_test, y_test

def sliding_window(list_dataset, sinalizador, window):
  X_train1, y_train1, X_test1, y_test1 = classification.generate_train_test(list_dataset[0], sinalizador)
  if window == 0: 
    for dataset in list_dataset:
      X_train2, y_train2, X_test2, y_test2 = classification.generate_train_test(dataset, sinalizador)
      X_train, y_train, X_test, y_test = concatenate_samples(X_train1, y_train1, X_test1, y_test1, X_train2, y_train2, X_test2, y_test2)
  else:
    for dataset in range(len(list_dataset)):
      if dataset != window:
        X_train2, y_train2, X_test2, y_test2 = classification.generate_train_test(list_dataset[dataset], sinalizador)
        X_train, y_train, X_test, y_test = concatenate_samples(X_train1, y_train1, X_test1, y_test1, X_train2, y_train2, X_test2, y_test2)

  X_train, y_train = shuffle(X_train, y_train)
  X_test, y_test = shuffle(X_test[100:], y_test[100:])
  return X_train, y_train, X_test, y_test


def experiment(list_dataset, list_names_dataset, list_sinalizadores,database_path,sliding):
  execute("CREATE TABLE IF NOT EXISTS results(name_model TEXT, dataset TEXT, accuracy FLOAT, precision FLOAT, recall FLOAT, f1 FLOAT)",database_path)
  if sliding:
    list_window = np.arange(0,len(list_dataset)).tolist()
  else:
    list_window = [0]
  for sinalizador in list_sinalizadores:
    for window in list_window:
      X_train, y_train, X_test, y_test = sliding_window(list_dataset, sinalizador, window)
      accuracy, precision, recall, f1 = individual(X_train, y_train, X_test, y_test)
      execute_insert("INSERT INTO results VALUES(?, ?, ?, ?, ?, ?)",('individual', list_names_dataset[window], accuracy, precision, recall, f1),database_path)
  return


def individual(X_train, y_train, X_test, y_test):
  from tcn import compiled_tcn
  from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
  model = compiled_tcn(return_sequences=False,num_feat=150,num_classes=20,nb_filters=32,kernel_size=5,dilations=[2 ** i for i in range(2)],
                       padding='causal',dropout_rate=0,use_batch_norm=True,nb_stacks=1,max_len=X_train[0:1].shape[1],opt='adam',
                       use_skip_connections=True)
  history = model.fit(X_train, y_train, epochs=100, workers=4, use_multiprocessing=True, verbose=0)
  yhat = model.predict(X_test).squeeze().argmax(axis=1)
  accuracy = model.evaluate(X_test, y_test)[1]
  precision = precision_score(y_test, yhat, average='macro')
  recall = recall_score(y_test, yhat, average='macro')
  f1 = f1_score(y_test, yhat, average='macro')
  return accuracy, precision, recall, f1
