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

def generate_train(train_X, matriz_path, lista):
    """
    Gera dados de treino para a CNN4
    :parametro train_X: dados de entrada de treino
    :return: dados de entrada e saída de treino
    """
    data = []
    labels = []
    idx=train_X
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

def generate_test(test_X, matriz_path, lista):
    """
    Gera dados de teste para a CNN4
    :parametro test_X: dados de entrada de teste
    :return: dados de entrada e saída de teste
    """
    data = []
    labels = []
    idx=test_X
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
          
    x_test = np.array(data, dtype = 'float32')
    lb = LabelBinarizer()
    y_test = lb.fit_transform(labels)
    y_test = np.stack((y_test,)*1, axis=-1)
    y_test = y_test.squeeze().argmax(axis=1)
    return x_test, y_test

def generate_train_test(matrix_path):
  indices, dados_Y, lista = return_labels(matrix_path)
  (train_X, test_X, train_Y, test_Y) = train_test_split(indices,dados_Y,random_state=42,test_size=0.25,stratify=dados_Y)
  X_train, y_train = generate_train(train_X, matrix_path, lista)
  X_test, y_test = generate_test(test_X, matrix_path, lista)
  return X_train, y_train, X_test, y_test
