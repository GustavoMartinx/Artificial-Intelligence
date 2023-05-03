from sklearn.preprocessing import MinMaxScaler
from collections import Counter
import numpy as np
import pandas as pd

def euclidian_distances(x_test, x_train):
    
    C = np.zeros(x_test.size)

    for i in range(x_test.size):
        
        C[i] = (x_test[i] - x_train[i])**2

    acc = np.sum(C)
    
    return (acc**(0.5))

class KNN:

    # Construtor
    def __init__(self, k):
        self.k = k

    # Treino
    def fit(self, X, Y):
        
        # armazenando amostras de treinamento
        self.X_train = X    # dados/features (normalizados)
        self.Y_train = Y    # labels/classes/última coluna

    # Novas amostras/Teste
    def predict(self, X):
        
        # X = Teste
        # x = linha do Teste (vet/lista com 132 elementos/caracteristicas)
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    # pega apenas uma amostra do conjunto (x minúsculo)
    def _predict(self, x):
        
        # obtendo as distâncias entre uma linha do teste e cada linha do treino
        distances = [euclidian_distances(x, x_train) for x_train in self.X_train]

        # ordena as distancias encontradas, as ordena
        # e seleciona os indices das K primeiras amostras
        k_indices = np.argsort(distances)[:self.k]

        # pega as classes as quais pertencem os k primeiros 
        k_nearest_labels = [self.Y_train[i] for i in k_indices]

        # mais votado, most commom class label
        most_commom = Counter(k_nearest_labels).most_common(1)
        return most_commom[0][0]
    



''' Abrindo/Carregando os aquivos '''

df_treino = pd.read_csv("features_prof/treinamento.txt", delimiter=' ')
df_teste = pd.read_csv("features_prof/teste.txt", delimiter=' ')

# print(df_treino)
# print(df_treino.describe())
# print(df_teste)
# print(df_teste.describe())


''' Separando os dados/features e as classes '''

# seleciona todo o Treino exceto a coluna das classes (última col) | X
features_treino = df_treino.iloc[:, :-1].values

# seleciona apenas a coluna da classe (última) | Y
classes_treino = df_treino.iloc[:, -1].values

# print(features_treino)
# print(classes_treino)

# seleciona todo o Teste exceto a coluna das classes (última col) | X
features_teste = df_teste.iloc[:, :-1].values

# seleciona apenas a coluna da classe (última) | Y
classes_teste = df_teste.iloc[:, -1].values

# print(features_teste)
# print(classes_teste)


''' Normalizando os dados '''

# criando o objeto normalizer
normalizer = MinMaxScaler()

# aplicando a normalização nos dados do Treino
features_train_normalized = normalizer.fit_transform(features_treino)

# aplicando a normalização nos dados do Teste
features_test_normalized = normalizer.fit_transform(features_teste)



''' Chamando KNN '''

classifier = KNN(k=1)

# Treino: passa os dados normalizados e suas classes pra func fit
classifier.fit(features_train_normalized, classes_treino)

# executa o teste
predictions = classifier.predict(features_test_normalized)

# calculando a acurácia
acc = np.sum(predictions == classes_teste) / len(classes_teste)
print(acc)
