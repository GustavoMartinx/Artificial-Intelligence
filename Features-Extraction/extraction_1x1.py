from sklearn.preprocessing import MinMaxScaler
from PIL import Image
import pandas as pd
import numpy as np
import os

class Extraction():
    
    blackPixels = 0
    whitePixels = 0

    # Criando arquivo txt
    with open("Treino_1x1.csv", "a") as res_Treino_1x1:

        # Current Working Directory
        project = os.getcwd()

        # percorre pelas pastas de numeros (classes) dentro de treino
        dirTreino = project + "/treino/"

        # (k^2) * 2 | 1x1 = 1 | 2x2 = 7 | 3x3 = 29 | 5x5 = 49
        # Col1, Col2, Col3
        # black,white,class
        cols = ""
        cols = cols + "Col1,Col2,Col3" + "\n"
        res_Treino_1x1.write(cols)

        for j in range(10):
            numberDir = dirTreino + str(j)

            # iterate over files in that directory
            for filename in os.scandir(numberDir):
                if filename.is_file():
                
                    # logica a ser aplicada a cada arquivo aqui
                    
                    # abrindo imagem
                    img = Image.open(filename.path) # "./treino/0/37138-2-4.bmp"

                    # obtendo as dimensões de uma imagem
                    imgWidth = img.size[0]      # largura
                    imgHeight = img.size[1]     # altura

                    # print("Inteira 1x1 imgWidth: ", imgWidth)
                    # print("Inteira 1x1 imgHeight: ", imgHeight)
                    
                    # percorrer por todos os pixels de uma imagem (quadrante 1x1)
                    for line in range(imgHeight):
                        for column in range(imgWidth):
                            
                            # using getpixel method
                            pixelValue = img.getpixel((column, line))

                            # se o pixel for preto
                            if pixelValue == 0:
                                blackPixels += 1
                            elif pixelValue == 255:
                                whitePixels += 1
                    
                    # printando a qtde de pixels B e W da imagem toda (1x1)
                    res_Treino_1x1.write(f"{blackPixels},{whitePixels},{str(j)}\n")
                    blackPixels = 0
                    whitePixels = 0


    # Normalizando dados de Treino
    table = pd.read_csv("Treino_1x1.csv")

    # print(table)
    # print(table.describe())

    features = table.iloc[:, :-1].values
    classes = table.iloc[:, -1].values

    # print(features)
    # print(classes)

    # Criando o objeto normalizer
    normalizer = MinMaxScaler()

    # Ajustando e aplicando a transformação
    minmax = normalizer.fit_transform(features)

    # Concatenando o min-max com a classe
    data_minmax = np.hstack((minmax, classes.reshape(-1, 1)))

    # print("-----")
    # print(data_minmax)

    # Salvando a base de dados normalizada
    np.savetxt('Treino_1x1_normalizado.txt', data_minmax, delimiter=',', header=cols, fmt='%f', comments='')




    # Teste
    with open("Teste_1x1.csv", "a") as res_Teste_1x1:

        # percorre pelas pastas de numeros dentro de teste
        dirTeste = project + "/teste/"

        # (k^2) * 2 | 1x1 = 1 | 2x2 = 7 | 3x3 = 29 | 5x5 = 49
        # Col1, Col2, Col3
        # black,white,class
        cols = ""
        cols = cols + "Col1,Col2,Col3" + "\n"
        res_Teste_1x1.write(cols)

        for i in range(10):
            numberDir = dirTeste + str(i)

            # iterate over files in that directory
            for filename in os.scandir(numberDir):
                if filename.is_file():
                    # logica a ser aplicada a cada arquivo aqui
                    
                    # abrindo imagem
                    img = Image.open(filename.path) # "./treino/0/37138-2-4.bmp"

                    # obtendo as dimensões de uma imagem
                    imgWidth = img.size[0]      # largura
                    imgHeight = img.size[1]     # altura

                    # print("Inteira 1x1 imgWidth: ", imgWidth)
                    # print("Inteira 1x1 imgHeight: ", imgHeight)
                    
                    # percorrer por todos os pixels de uma imagem (quadrante 1x1)
                    for line in range(imgHeight):
                        for column in range(imgWidth):
                            
                            # using getpixel method
                            pixelValue = img.getpixel((column, line))

                            # se o pixel for preto
                            if pixelValue == 0:
                                blackPixels += 1
                            elif pixelValue == 255:
                                whitePixels += 1
                    
                    # printando a qtde de pixels B e W da imagem toda (1x1)
                    res_Teste_1x1.write(f"{blackPixels},{whitePixels},{str(i)}\n")
                    blackPixels = 0
                    whitePixels = 0
                    
    # Normalizando dados de Teste
    df_teste = pd.read_csv("Teste_1x1.csv")

    # print(df_teste)
    # print(df_teste.describe())

    features = df_teste.iloc[:, :-1].values
    classes = df_teste.iloc[:, -1].values

    # print(features)
    # print(classes)

    # Criando o objeto normalizer
    normalizer = MinMaxScaler()

    # Ajustando e aplicando a transformação
    minmax = normalizer.fit_transform(features)

    # Concatenando o min-max com a classe
    data_minmax = np.hstack((minmax, classes.reshape(-1, 1)))

    # print("-----")
    # print(data_minmax)

    # Salvando a base de dados normalizada
    np.savetxt('Teste_1x1_normalizado.txt', data_minmax, delimiter=',', header=cols, fmt='%f', comments='')



"""
2) para avaliar se o numero de pixels das diferentes imagens da mesma classe são similiares:

- fazer a normalização dos valores dos pixels pretos e brancos de cada classe (cada número)
- estabelecer um range a partir do valor normalizado
- intervalo este que, se o valor do pixel estiver dentro, será considerado similar, caso contrário não 


3) normalizar os dados (QUAIS DADOS??) usando minmax ou zscore


5) dividir cada imagem em quadrantes e contar a qtde de pixels de cada quadrante

- num preto e branco para:
- cada quadrante (1x1 - img normal), 2x2 - quatro quadrados, 3x3 - nove quadrados, 5x5)
- cada imagem
- cada classe
- treino e teste

"""