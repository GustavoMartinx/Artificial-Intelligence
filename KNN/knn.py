from sklearn.preprocessing import MinMaxScaler
from collections import Counter
import numpy as np
import pandas as pd

def euclidian_distances(x_test, x_train):
    
    C = x_test - x_train
    C = np.square(C)
    acc = np.sum(C)
    return (acc**(0.5))



# Retorna um novo array com X linhas selecionadas aleatoriamente do array de entrada
def sample_data(array, X):

    # Embaralhar as linhas do array
    np.random.shuffle(array)

    # calcula a quantidade de linhas a serem selecionadas
    n_rows = int(len(array) * X)

    # seleciona as n_rows primeiras linhas do array embaralhado
    return array[:n_rows]


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
        
        # obtém as distâncias entre uma linha do Teste e cada linha do Treino
        distances = [euclidian_distances(x, x_train) for x_train in self.X_train]

        # ordena as distâncias encontradas, e seleciona
        # os indices das K primeiras amostras
        k_indices = np.argsort(distances)[:self.k]

        # pega as classes as quais pertencem os k primeiros
        k_nearest_labels = [self.Y_train[i] for i in k_indices]

        # obtém classe mais votado, most commom class label
        most_commom = Counter(k_nearest_labels).most_common(1)
        return most_commom[0][0]
    



''' Abrindo/Carregando os aquivos '''

df_treino = pd.read_csv("features_prof/treinamento.txt", delimiter=' ')
df_teste = pd.read_csv("features_prof/teste.txt", delimiter=' ')


''' Separando os dados/features e as classes '''

# seleciona todo o Treino exceto a coluna das classes (última col) | X
features_treino = df_treino.iloc[:, :-1].values

# seleciona apenas a coluna da classe (última) | Y
classes_treino = df_treino.iloc[:, -1].values


# seleciona todo o Teste exceto a coluna das classes (última col) | X
features_teste = df_teste.iloc[:, :-1].values

# seleciona apenas a coluna da classe (última) | Y
classes_teste = df_teste.iloc[:, -1].values



''' Separando os dados em 25% e 50% '''

# features_treino_25 = sample_data(features_treino, 0.25)
# features_treino_50 = sample_data(features_treino, 0.5)



''' Normalizando os dados 

# criando o objeto normalizer
normalizer = MinMaxScaler()

# aplicando a normalização nos dados do Treino
features_train_normalized = normalizer.fit_transform(features_treino)

# aplicando a normalização nos dados do Teste
features_test_normalized = normalizer.fit_transform(features_teste)
'''


''' Chamando KNN '''
K = 1
classifier = KNN(k=K)

# Treino: passa os dados e suas classes pra func fit
classifier.fit(features_treino, classes_treino) # features_treino = 100% | features_treino_50 = 50% | features_treino_25 = 25%

# executa o teste
predictions = classifier.predict(features_teste)

# calculando a acurácia
acc = np.sum(predictions == classes_teste) / len(classes_teste)
print(f"k = {K}: {round(acc*100, 3)}%")
