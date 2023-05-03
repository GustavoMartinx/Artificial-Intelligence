import numpy as np
from collections import Counter

def euclidian_distances(x_test, x_train):
    
    C = [x_test.size]

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
        self.X_train = X
        self.Y_train = Y

    # Novas amostras/Teste
    def predict(self, X, Y):
        
        # X = Teste
        # x = linha (lista 132 caracteristicas) do Teste
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    # pega apenas uma amostra do conjunto (x minúsculo)
    def _predict(self, x):
        
        # obtendo as distâncias
        distances = [euclidian_distances(x, x_train) for x_train in self.X_train]

        # ordena as distancias encontradas, as ordena
        # e seleciona os indices das K primeiras amostras
        k_indices = np.argsort(distances)[:self.k]

        # pega as classes as quais pertencem os k primeiros 
        k_nearest_labels = [self.Y_train[i] for i in k_indices]

        # mais votado, most commom class label
        most_commom = Counter(k_nearest_labels).most_common(1)
        return most_commom[0][0]
